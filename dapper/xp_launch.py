"""Tools (notably `xpList`) for setup and running of experiments (known as `xp`s).

See `dapper.da_methods.da_method` for the strict definition of `xp`s.
"""

import copy
import dataclasses as dcs
import os
import re
import shutil
import sys
from datetime import datetime
from functools import wraps
from pathlib import Path
from textwrap import dedent

import dill
import numpy as np
import struct_tools
import tabulate as _tabulate
from tabulate import tabulate
from tqdm.auto import tqdm

import dapper.stats
import dapper.tools.progressbar as pb
from dapper.dpr_config import rc
from dapper.tools.remote.uplink import submit_job_GCP
from dapper.tools.seeding import set_seed

_tabulate.MIN_PADDING = 0
XP_TIMESTAMP_TEMPLATE = "run_%Y-%m-%d__%H-%M-%S"


def seed_and_simulate(HMM, xp):
    """Default experiment setup (sets seed and simulates truth and obs).

    Used by `xpList.launch` via `run_experiment`.

    Parameters
    ----------
    HMM: HiddenMarkovModel
        Container defining the system.
    xp: object
        Type: a `dapper.da_methods.da_method`-decorated class.

        .. note:: `xp.seed` should be an integer. Otherwise:
            If there is no `xp.seed` then then the seed is not set.
            Although different `xp`s will then use different seeds
            (unless you do some funky hacking),
            reproducibility for your script as a whole would still be obtained
            by setting the seed at the outset (i.e. in the script).
            On the other hand, if `xp.seed in [None, "clock"]`
            then the seed is from the clock (for each xp),
            which would not provide exact reproducibility.

    Returns
    -------
    tuple (xx, yy)
        The simulated truth and observations.
    """
    set_seed(getattr(xp, 'seed', False))
    xx, yy = HMM.simulate()
    return HMM, xx, yy


def run_experiment(xp, label, savedir, HMM, setup=seed_and_simulate, free=True,
                   statkeys=False, fail_gently=False, **stat_kwargs):
    """Used by `xpList.launch` to run each single (DA) experiment ("xp").

    This involves steps similar to `examples/basic_1.py`, i.e.:

    - `setup`                    : Initialize experiment.
    - `xp.assimilate`            : run DA, pass on exception if fail_gently
    - `xp.stats.average_in_time` : result averaging
    - `xp.avrgs.tabulate`        : result printing
    - `dill.dump`                : result storage

    Parameters
    ----------
    xp: object
        Type: a `dapper.da_methods.da_method`-decorated class.
    label: str
        Name attached to progressbar during assimilation.
    savedir: str
        Path of folder wherein to store the experiment data.
    HMM: HiddenMarkovModel
        Container defining the system.
    setup: function
        This function must take two arguments: `HMM` and `xp`,
        and return the `HMM` to be used by the DA methods
        (typically the same as the input `HMM`, but could be modified),
        and the (typically synthetic) truth and obs time series.

        This gives you the ability to customize almost any aspect of the
        individual experiments within a batch launch of experiments.
        Typically you will grab one or more parameter values
        stored in the `xp` (see `dapper.da_methods.da_method`) and act on them,
        or set them in some other object that impacts the experiment.
        Thus, by generating a new `xp` for each such parameter value you can
        investigate the impact/sensitivity of the results to this parameter.
        Examples include:

        - Setting the seed. See the default `setup`, namely `seed_and_simulate`,
          for how this is done.
        - Setting some aspect of the `HMM` such as the observation noise,
          or the interval between observations.
        - Setting some parameter of the model (not otherwise detailed in the `HMM`).
          For example, the `Force` parameter of `dapper.mods.Lorenz96`, as done in
          see `examples/basic_3`.
        - Using a different `HMM` entirely for the truth/obs (`xx`/`yy`) generation,
          than the one that will be used by the DA. Or loading the truth/obs
          time series from file. In both cases, you might also have to do some
          cropping or slicing of `xx` and `yy` before returning them.
    free: bool
        Whether (or not) to `del xp.stats` after the experiment is done,
        so as to free up memory and/or not save this data
        (just keeping `xp.avrgs`).
    statkeys: list
        A list of names (possibly in the form of abbreviations) of the
        statistical averages that should be printed immediately afther
        this xp.
    fail_gently: bool
        Whether (or not) to propagate exceptions.
    """
    # We should copy HMM so as not to cause any nasty surprises such as
    # expecting param=1 when param=2 (coz it's not been reset).
    # NB: won't copy implicitly ref'd obj's (like L96's core). => bug w/ MP?
    hmm = copy.deepcopy(HMM)

    # GENERATE TRUTH/OBS
    hmm, xx, yy = setup(hmm, xp)

    # ASSIMILATE
    xp.assimilate(hmm, xx, yy, label, fail_gently=fail_gently, **stat_kwargs)

    # Clear references to mpl (for pickling purposes)
    if hasattr(xp.stats, "LP_instance"):
        del xp.stats.LP_instance

    # AVERAGE
    xp.stats.average_in_time(free=free)

    # PRINT
    if statkeys:
        statkeys = () if statkeys is True else statkeys
        print(xp.avrgs.tabulate(statkeys))

    # SAVE
    if savedir:
        with open(Path(savedir)/"xp", "wb") as FILE:
            dill.dump({'xp': xp}, FILE)


def collapse_str(string, length=6):
    """Truncate a string (in the middle) to the given `length`"""
    if len(string) <= length:
        return string
    else:
        return string[:length-2]+'~'+string[-1]


class xpList(list):
    """Subclass of `list` specialized for experiment ("xp") objects.

    Main use: administrate experiment launches.

    Modifications to `list`:

    - `xpList.append` supports `unique` to enable lazy `xp` declaration.
    - `__iadd__` (`+=`) supports adding single `xp`s.
      this is hackey, but convenience is king.
    - `__getitem__` supports lists, similar to `np.ndarray`
    - `__repr__`: prints the list as rows of a table,
      where the columns represent attributes whose value is not shared among all `xp`s.
      Refer to `xpList.split_attrs` for more information.

    Add-ons:

    - `xpList.launch`: run the experiments in current list.
    - `xpList.split_attrs`: find all attributes of the `xp`s in the list;
      classify as distinct, redundant, or common.
    - `xpList.gen_names`: use `xpList.split_attrs` to generate
      a short & unique name for each `xp` in the list.
    - `xpList.tabulate_avrgs`: tabulate time-averaged results.
    - `xpList.inds` to search by kw-attrs.

    Parameters
    ----------
    args: entries
        Nothing, or a list of `xp`s.

    unique: bool
        Duplicates won't get appended. Makes `append` (and `__iadd__`) relatively slow.
        Use `extend` or `__add__` or `combinator` to bypass this validation.

    Also see
    --------
    - Examples: `examples/basic_2`, `examples/basic_3`
    - `dapper.xp_process.xpSpace`, which is used for experient result **presentation**,
      as opposed to this class (`xpList`), which handles **launching** experiments.
    """

    def __init__(self, *args, unique=False):
        self.unique = unique
        super().__init__(*args)

    def __iadd__(self, xp):
        if not hasattr(xp, '__iter__'):
            xp = [xp]
        for item in xp:
            self.append(item)
        return self

    def append(self, xp):
        """Append **if** not `self.unique` & present."""
        if not (self.unique and xp in self):
            super().append(xp)

    def __getitem__(self, keys):
        """Indexing, also by a list"""
        try:
            B = [self[k] for k in keys]    # if keys is list
        except TypeError:
            B = super().__getitem__(keys)  # if keys is int, slice
        if hasattr(B, '__len__'):
            B = xpList(B)                  # Cast
        return B

    def inds(self, strict=True, missingval="NONSENSE", **kws):
        """Find (all) indices of `xps` whose attributes match kws.

        If strict, then `xp`s lacking a requested attr will not match,
        unless the missingval (e.g. `None`) matches the required value.
        """
        def match(xp):
            def missing(v): return missingval if strict else v
            matches = [getattr(xp, k, missing(v)) == v for k, v in kws.items()]
            return all(matches)

        return [i for i, xp in enumerate(self) if match(xp)]

    @property
    def da_methods(self):
        """List `da_method` attributes in this list."""
        return [xp.da_method for xp in self]

    def split_attrs(self, nomerge=()):
        """Classify each/every attribute of the `xp`s into: distinct, redundant, common.

        Insert `None` if an attribute is distinct but not in `xp`.

        Used by `repr` on `xpList` to make a nice table of the list,
        in which only the attributes that differ among the `xp`s
        need be included.

        Parameters
        ----------
        nomerge: list
            Attributes that should always be seen as distinct.
        """
        def _aggregate_keys():
            """Aggregate keys from all `xp`"""
            if len(self) == 0:
                return []

            # Start with da_method
            aggregate = ['da_method']

            # Aggregate all other keys
            for xp in self:

                # Get dataclass fields
                try:
                    dc_fields = dcs.fields(xp.__class__)
                    dc_names = [F.name for F in dc_fields]
                    keys = xp.__dict__.keys()
                except TypeError:
                    # Assume namedtuple
                    dc_names = []
                    keys = xp._fields

                # For all potential keys:
                for k in keys:
                    # If not already present:
                    if k not in aggregate:

                        # If dataclass, check repr:
                        if k in dc_names:
                            if dc_fields[dc_names.index(k)].repr:
                                aggregate.append(k)
                        # Else, just append
                        else:
                            aggregate.append(k)

            # Remove unwanted
            excluded  = [re.compile('^_'), 'avrgs', 'stats', 'HMM', 'duration']
            aggregate = struct_tools.complement(aggregate, excluded)
            return aggregate

        distinct, redundant, common = {}, {}, {}

        for key in _aggregate_keys():

            def _getattr(xp, key):
                # Don't use None, to avoid mixing with actual None's
                # TODO 4: use an object yet more likely to be unique.
                missing = "N/A"
                a = getattr(xp, key, missing)

                # Replace ndarray by its id, since o/w it will
                # complain that you must use all().
                # Alternative: replace all == (and !=) below by "is".
                #     Tabulation with multi-line params actually works,
                #     (though it's still likely to take up too much space,
                #     unless we set np.printoptions...).
                #     However, then python (since 3.8) will complain about
                #     comparison to literal.
                if isinstance(a, np.ndarray):
                    shorten = 6
                    a = f"arr(<id {id(a)//10**shorten}>)"
                # TODO 3: leave formatting to sub() below?
                # TODO 4: do similar formatting for other non-trivial params?
                # TODO 4: document alternative way to specify non-trivial params:
                #         use key to be looked up in some globally accessible dct.
                #         Advantage: names are meaningful, rather than ids.
                return a

            vals = [_getattr(xp, key) for xp in self]

            # Sort (assign dct) into distinct, redundant, common
            if struct_tools.flexcomp(key, *nomerge):
                # nomerge => Distinct
                dct, vals = distinct, vals
            elif all(vals[0] == v for v in vals):
                # all values equal => common
                dct, vals = common, vals[0]
            else:
                v0 = next(v for v in vals if "N/A" != v)
                if all(v == "N/A" or v == v0 for v in vals):
                    # all values equal or "N/A" => redundant
                    dct, vals = redundant, v0
                else:
                    # otherwise => distinct
                    dct, vals = distinct, vals

            # Replace "N/A" by None
            def sub(v): return None if v == "N/A" else v
            if isinstance(vals, str):
                vals = sub(vals)
            else:
                try:
                    vals = [sub(v) for v in vals]
                except TypeError:
                    vals = sub(vals)

            dct[key] = vals

        return distinct, redundant, common

    def __repr__(self):
        distinct, redundant, common = self.split_attrs()
        s = '<xpList> of length %d with attributes:\n' % len(self)
        s += tabulate(distinct, headers="keys", showindex=True)
        s += "\nOther attributes:\n"
        s += str(struct_tools.AlignedDict({**redundant, **common}))
        return s

    def gen_names(self, abbrev=6, tab=False):
        """Similiar to `self.__repr__()`, but:

        - returns *list* of names
        - tabulation is optional
        - attaches (abbreviated) labels to each attribute
        """
        distinct, redundant, common = self.split_attrs(nomerge=["da_method"])
        labels = distinct.keys()
        values = distinct.values()

        # Label abbreviation
        labels = [collapse_str(k, abbrev) for k in labels]

        # Make label columns: insert None or lbl+":", depending on value
        def column(lbl, vals):
            return [None if v is None else lbl+":" for v in vals]
        labels = [column(lbl, vals) for lbl, vals in zip(labels, values)]

        # Interlace labels and values
        table = [x for (a, b) in zip(labels, values) for x in (a, b)]

        # Rm da_method label (but keep value)
        table.pop(0)

        # Transpose
        table = list(map(list, zip(*table)))

        # Tabulate
        table = tabulate(table, tablefmt="plain")

        # Rm space between lbls/vals
        table = re.sub(':  +', ':', table)

        # Rm alignment
        if not tab:
            table = re.sub(r' +', r' ', table)

        return table.splitlines()

    @wraps(dapper.stats.tabulate_avrgs)
    def tabulate_avrgs(self, *args, **kwargs):
        distinct, redundant, common = self.split_attrs()
        averages = dapper.stats.tabulate_avrgs([C.avrgs for C in self], *args, **kwargs)
        columns = {**distinct, '|': ['|']*len(self), **averages}  # merge
        return tabulate(columns, headers="keys", showindex=True).replace('␣', ' ')

    def launch(self, HMM, save_as="noname", mp=False, fail_gently=None, **kwargs):
        """Essentially: `for xp in self: run_experiment(xp, ..., **kwargs)`.

        See `run_experiment` for documentation on the `kwargs` and `fail_gently`.

        The results are saved in `rc.dirs.data / save_as`,
        unless `save_as` is `False`/`None`.
        Note: multiprocessing (locally or in the cloud) requires saving/loading data.

        Depending on `mp`, `run_experiment` is delegated as follows:

        - `False`: caller process (no parallelisation)
        - `True` or `"MP"` or an `int`: multiprocessing on this host
        - `"GCP"` or `"Google"` or `dict(server="GCP")`: the DAPPER server
          (Google Cloud Computing with HTCondor).
            - Specify a list of files as `mp["files"]` to include them
              in working directory of the server workers.
            - In order to use absolute paths, the list should cosist
              of tuples, where the first item is relative to the second
              (which is an absolute path). The root is then not included
              in the working directory of the server.
            - If this dict field is empty, then all python files
              in `sys.path[0]` are uploaded.

        See `examples/basic_2.py` and `examples/basic_3.py` for example use.
        """
        # Parse mp option
        if not mp:
            mp = dict()
        elif mp in [True, "MP"]:
            mp = dict(server="local")
        elif isinstance(mp, int):
            mp = dict(server="local", NPROC=mp)
        elif mp in ["GCP", "Google"]:
            mp = dict(server="GCP", files=[], code="")

        # Parse fail_gently
        if fail_gently is None:
            if mp and mp["server"] == "GCP":
                fail_gently = False
                # coz cloud processing is entirely de-coupled anyways
            else:
                fail_gently = True
                # True unless otherwise requested
        kwargs["fail_gently"] = fail_gently

        # Bundle HMM with kwargs
        kwargs['HMM'] = HMM

        # Parse save_as
        if save_as in [None, False]:
            assert not mp, "Multiprocessing requires saving data."
            # Parallelization w/o storing is possible, especially w/ threads.
            # But it involves more complicated communication set-up.
            def xpi_dir(*args): return None
        else:
            save_as = rc.dirs.data / Path(save_as).stem
            save_as /= datetime.now().strftime(XP_TIMESTAMP_TEMPLATE)
            os.makedirs(save_as)
            print(f"Experiment stored at {save_as}")

            def xpi_dir(i):
                path = save_as / str(i)
                os.mkdir(path)
                return path

        # No parallelization
        if not mp:
            for ixp, (xp, label) in enumerate(zip(self, self.gen_names())):
                run_experiment(xp, label, xpi_dir(ixp), **kwargs)

        # Local multiprocessing
        elif mp["server"].lower() == "local":
            def run_with_fixed_args(arg):
                xp, ixp = arg
                run_experiment(xp, None, xpi_dir(ixp), **kwargs)
            args = zip(self, range(len(self)))

            pb.disable_progbar          = True
            pb.disable_user_interaction = True
            NPROC = mp.get("NPROC", None)  # None => mp.cpu_count()
            from dapper.tools.multiproc import mpd  # will fail on GCP
            with mpd.Pool(NPROC) as pool:
                list(tqdm(
                    pool.imap(
                        run_with_fixed_args, args),
                    total=len(self),
                    desc="Parallel experim's",
                    smoothing=0.1))
            pb.disable_progbar          = False
            pb.disable_user_interaction = False

        # Google cloud platform, multiprocessing
        elif mp["server"] == "GCP":
            for ixp, xp in enumerate(self):
                with open(xpi_dir(ixp)/"xp.var", "wb") as f:
                    dill.dump(dict(xp=xp), f)

            with open(save_as/"xp.com", "wb") as f:
                dill.dump(kwargs, f)

            # mkdir extra_files
            extra_files = save_as / "extra_files"
            os.mkdir(extra_files)
            # Default files: .py files in sys.path[0] (main script's path)
            if not mp.get("files", []):
                ff = os.listdir(sys.path[0])
                mp["files"] = [f for f in ff if f.endswith(".py")]
            # Copy files into extra_files
            for f in mp["files"]:
                if isinstance(f, (str, Path)):
                    # Example: f = "A.py"
                    path = Path(sys.path[0]) / f
                    dst = f
                else:  # instance of tuple(path, root)
                    # Example: f = ("~/E/G/A.py", "G")
                    path, root = f
                    dst = Path(path).relative_to(root)
                dst = extra_files / dst
                os.makedirs(dst.parent, exist_ok=True)
                try:
                    shutil.copytree(path, dst)  # dir -r
                except OSError:
                    shutil.copy2(path, dst)  # file

            # Loads PWD/xp_{var,com} and calls run_experiment()
            with open(extra_files/"load_and_run.py", "w") as f:
                f.write(dedent("""\
                import dill
                from dapper.xp_launch import run_experiment

                # Load
                with open("xp.com", "rb") as f: com = dill.load(f)
                with open("xp.var", "rb") as f: var = dill.load(f)

                # User-defined code
                %s

                # Run
                result = run_experiment(var['xp'], None, ".", **com)
                """) % dedent(mp["code"]))

            with open(extra_files/"dpr_config.yaml", "w") as f:
                f.write("\n".join([
                    "data_root: '$cwd'",
                    "liveplotting: no",
                    "welcome_message: no"]))
            submit_job_GCP(save_as)

        return save_as


def combinator(param_dict, **glob_dict):
    """Mass creation of `xp`'s by combining the value lists in the `param_dict`.

    Returns a function (`for_params`) that creates all possible combinations
    of parameters (from their value list) for a given `dapper.da_methods.da_method`.
    This is a good deal more efficient than relying on `xpList`'s `unique`. Parameters

    - not found among the args of the given DA method are ignored by `for_params`.
    - specified as keywords to the `for_params` fix the value
      preventing using the corresponding (if any) value list in the `param_dict`.

    .. caution::
        Beware! If, eg., `infl` or `rot` are in `param_dict`, aimed at the `EnKF`,
        but you forget that they are also attributes some method where you don't
        actually want to use them (eg. `SVGDF`),
        then you'll create many more than you intend.
    """
    def for_params(method, **fixed_params):
        dc_fields = [f.name for f in dcs.fields(method)]
        params = struct_tools.intersect(param_dict, dc_fields)
        params = struct_tools.complement(params, fixed_params)
        params = {**glob_dict, **params}  # glob_dict 1st

        def xp1(dct):
            xp = method(**struct_tools.intersect(dct, dc_fields), **fixed_params)
            for key, v in struct_tools.intersect(dct, glob_dict).items():
                setattr(xp, key, v)
            return xp

        return [xp1(dct) for dct in struct_tools.prodct(params)]
    return for_params

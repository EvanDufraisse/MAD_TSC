import collections
import itertools
import random
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Union
import warnings

from optuna.distributions import BaseDistribution
from optuna.logging import get_logger
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
import time
import os

from regex import E
import datetime
import dateparser
import sqlite3


GridValueType = Union[str, float, int, bool, None]
SortableParamValueSequenceType = Union[List[str], List[float], List[int], List[bool]]


_logger = get_logger(__name__)


class NoGridLeftToVisit(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


import datetime
from time import mktime


def add_seconds(d, dela_seconds):
    parsed = dateparser.parse(d)
    new_date = parsed + datetime.timedelta(seconds=dela_seconds)
    new_date_str = new_date.strftime("%Y-%m-%d %H:%M:%S")
    return new_date_str


def get_last_time(conn):
    # conn.execute("BEGIN EXCLUSIVE")
    r = conn.execute("""SELECT * FROM time""")
    fetched = r.fetchall()
    return fetched[0][0]


def get_last_time_and_update(path_lock_db, delta_seconds=10):
    conn = sqlite3.connect(path_lock_db)
    conn.execute("BEGIN EXCLUSIVE")
    r = conn.execute("""SELECT * FROM time""")
    fetched = r.fetchall()
    last_time = fetched[0][0]
    new_time = add_seconds(last_time, delta_seconds)
    value_time = (new_time,)
    conn.execute("""UPDATE time SET time = ?""", value_time)
    conn.execute("END")
    conn.close()
    return last_time, new_time


def update_time(path_lock_db, d):
    value_time = (d,)
    conn = sqlite3.connect(path_lock_db)
    conn.execute("BEGIN EXCLUSIVE")
    conn.execute("""UPDATE time SET time = ?""", value_time)
    conn.execute("END")
    conn.close()


class MultiprocessLockManager(object):
    def __init__(self, path_lock_db, time_before_failure=320):
        self.path_lock_db = path_lock_db
        self.conn = sqlite3.connect(self.path_lock_db)
        init_time = time.time()
        success = False
        while not success:
            if time.time() - init_time > time_before_failure:
                raise TimeoutError(
                    f"Could not get lock on database {self.path_lock_db} after {time_before_failure} seconds"
                )
            try:
                self.conn.execute("BEGIN EXCLUSIVE")
                success = True
            except sqlite3.OperationalError:
                time.sleep(5)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.execute("END")
        self.conn.__exit__(exc_type, exc_val, exc_tb)

    def __del__(self):
        self.conn.close()


class CustomGridSampler(BaseSampler):
    """Sampler using grid search.

    With :class:`~optuna.samplers.GridSampler`, the trials suggest all combinations of parameters
    in the given search space during the study.

    Example:

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -100, 100)
                y = trial.suggest_int("y", -100, 100)
                return x ** 2 + y ** 2


            search_space = {"x": [-50, 0, 50], "y": [-99, 0, 99]}
            study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
            study.optimize(objective)

    Note:

        :class:`~optuna.samplers.GridSampler` automatically stops the optimization if all
        combinations in the passed ``search_space`` have already been evaluated, internally
        invoking the :func:`~optuna.study.Study.stop` method.

    Note:

        :class:`~optuna.samplers.GridSampler` does not take care of a parameter's quantization
        specified by discrete suggest methods but just samples one of values specified in the
        search space. E.g., in the following code snippet, either of ``-0.5`` or ``0.5`` is
        sampled as ``x`` instead of an integer point.

        .. testcode::

            import optuna


            def objective(trial):
                # The following suggest method specifies integer points between -5 and 5.
                x = trial.suggest_float("x", -5, 5, step=1)
                return x ** 2


            # Non-int points are specified in the grid.
            search_space = {"x": [-0.5, 0.5]}
            study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
            study.optimize(objective, n_trials=2)

    Note:
        A parameter configuration in the grid is not considered finished until its trial is
        finished. Therefore, during distributed optimization where trials run concurrently,
        different workers will occasionally suggest the same parameter configuration.
        The total number of actual trials may therefore exceed the size of the grid.

    Note:
        The grid is randomly shuffled and the order in which parameter configurations are
        suggested may vary. This is to reduce duplicate suggestions during distributed
        optimization.

    Args:
        search_space:
            A dictionary whose key and value are a parameter name and the corresponding candidates
            of values, respectively.
    """

    def __init__(
        self,
        search_space: Mapping[str, Sequence[GridValueType]],
        path_lock_db: str,
        max_fail_trials_per_grid=3,
        no_wait_mode=True,
    ) -> None:
        self.path_lock_db = path_lock_db
        if not os.path.exists(self.path_lock_db):
            with sqlite3.connect(self.path_lock_db) as conn:
                conn.execute("BEGIN EXCLUSIVE")
                # Create time database
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS time (last_time TIMESTAMP)"""
                )
                conn.execute("""INSERT INTO time VALUES (CURRENT_TIMESTAMP)""")
                conn.execute("END")
        for param_name, param_values in search_space.items():
            for value in param_values:
                self._check_value(param_name, value)

        self._search_space = collections.OrderedDict()
        for param_name, param_values in sorted(
            search_space.items(), key=lambda x: x[0]
        ):
            param_values = cast(SortableParamValueSequenceType, param_values)

            self._search_space[param_name] = sorted(param_values)

        self._all_grids = list(itertools.product(*self._search_space.values()))
        self._param_names = sorted(search_space.keys())
        self._n_min_trials = len(self._all_grids)
        self.grid_fail_counter = {}
        self.max_fail_trials_per_grid = max_fail_trials_per_grid
        self.time_trials = [0]
        self.last_time = -1
        self.no_wait_mode = no_wait_mode

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        with MultiprocessLockManager(self.path_lock_db) as lock_manager:
            # Instead of returning param values, GridSampler puts the target grid id as a system attr,
            # and the values are returned from `sample_independent`. This is because the distribution
            # object is hard to get at the beginning of trial, while we need the access to the object
            # to validate the sampled value.

            target_grids = self._get_unvisited_grid_ids(study, lock_manager)

            if len(target_grids) == 0:
                # This case may occur with distributed optimization or trial queue. If there is no
                # target grid, `GridSampler` evaluates a visited, duplicated point with the current
                # trial. After that, the optimization stops.

                # _logger.warning(
                #     "`GridSampler` is re-evaluating a configuration because the grid has been "
                #     "exhausted. This may happen due to a timing issue during distributed optimization "
                #     "or when re-running optimizations on already finished studies."
                # )

                # # One of all grids is randomly picked up in this case.
                # target_grids = list(range(len(self._all_grids)))
                # _logger.warning(
                #     "`GridSampler` hasn't found any unvisited grid. Stopping study."
                # )
                raise NoGridLeftToVisit(
                    "`GridSampler` hasn't found any unvisited grid."
                )

            # In distributed optimization, multiple workers may simultaneously pick up the same grid.
            # To make the conflict less frequent, the grid is chosen randomly.
            grid_id = random.choice(target_grids)

            study._storage.set_trial_system_attr(
                trial._trial_id, "search_space", self._search_space
            )
            study._storage.set_trial_system_attr(trial._trial_id, "grid_id", grid_id)

        return {}

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if param_name not in self._search_space:
            message = "The parameter name, {}, is not found in the given grid.".format(
                param_name
            )
            raise ValueError(message)

        # TODO(c-bata): Reduce the number of duplicated evaluations on multiple workers.
        # Current selection logic may evaluate the same parameters multiple times.
        # See https://gist.github.com/c-bata/f759f64becb24eea2040f4b2e3afce8f for details.
        grid_id = trial.system_attrs["grid_id"]
        param_value = self._all_grids[grid_id][self._param_names.index(param_name)]
        contains = param_distribution._contains(
            param_distribution.to_internal_repr(param_value)
        )
        if not contains:
            warnings.warn(
                f"The value `{param_value}` is out of range of the parameter `{param_name}`. "
                f"The value will be used but the actual distribution is: `{param_distribution}`."
            )

        return param_value

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        with MultiprocessLockManager(self.path_lock_db) as lock_manager:
            target_grids = self._get_unvisited_grid_ids(study, lock_manager)

            if len(target_grids) == 0:
                study.stop()
            elif len(target_grids) == 1:
                grid_id = study._storage.get_trial_system_attrs(trial._trial_id)[
                    "grid_id"
                ]
                if grid_id == target_grids[0]:
                    study.stop()

    @staticmethod
    def _check_value(param_name: str, param_value: Any) -> None:
        if param_value is None or isinstance(param_value, (str, int, float, bool)):
            return

        raise ValueError(
            "{} contains a value with the type of {}, which is not supported by "
            "`GridSampler`. Please make sure a value is `str`, `int`, `float`, `bool`"
            " or `None`.".format(param_name, type(param_value))
        )

    def get_time_left_slurm(self):
        # process = os.popen('squeue -h -j $SLURM_JOBID -o %L')
        # preprocessed = process.read()
        # process.close()
        # if preprocessed != '':
        #     preprocessed = preprocessed.strip()
        #     try:
        #     	day_time = int(preprocessed.split("-")[0])*24*3600
        #     except:
        #         day_time = 0
        #     hours = preprocessed.split("-")[-1].split(":")
        #     hours = [0]*(3*len(hours)) + hours
        #     hour_time = int(hours[0])*3600 + int(hours[1])*60 + int(hours[2])
        #     return day_time + hour_time
        # else:
        return 10000000000000

    def _get_unvisited_grid_ids(self, study: Study, lock_manager=None) -> List[int]:
        # List up unvisited grids based on already finished ones.
        running_grids = []
        unvisited_grids = set([])

        # We directly query the storage to get trials here instead of `study.get_trials`,
        # since some pruners such as `HyperbandPruner` use the study transformed
        # to filter trials. See https://github.com/optuna/optuna/issues/2327 for details.
        if not (lock_manager is None):
            last_time = get_last_time(lock_manager.conn)
            self.last_time = mktime(dateparser.parse(last_time).timetuple())

        trials = study._storage.get_all_trials(study._study_id, deepcopy=False)
        self.grid_fail_counter = {}

        if self.last_time == -1:
            pass
        else:
            try:
                self.time_trials.append(time.time() - self.last_time)
            except:
                print("time_trials error")
                print(self.last_time)
                msg = f"time_trials error {self.last_time}"
                raise Exception(msg)

        loop = False
        while not (loop) or (len(running_grids) > 1 and len(unvisited_grids) == 0):
            visited_grids = []
            definitively_failed_grids = []
            running_grids = []
            if loop:
                if self.no_wait_mode:
                    print("Looping no wait for current runs, exiting")
                    return []
                time.sleep(10)
            if self.get_time_left_slurm() < max(self.time_trials):
                print(f"time left job is {self.get_time_left_slurm()}")
                print(f"max time trial is {max(self.time_trials)}")
                return []

            for t in trials:
                if "grid_id" in t.system_attrs and self._same_search_space(
                    t.system_attrs["search_space"]
                ):
                    if t.state.is_finished() and not (t.state == TrialState.FAIL):
                        visited_grids.append(t.system_attrs["grid_id"])
                    elif t.state == TrialState.RUNNING:
                        running_grids.append(t.system_attrs["grid_id"])

                    if t.state == TrialState.FAIL:
                        if t.system_attrs["grid_id"] in self.grid_fail_counter:
                            self.grid_fail_counter[t.system_attrs["grid_id"]] += 1
                        else:
                            self.grid_fail_counter[t.system_attrs["grid_id"]] = 1
                    definitively_failed_grids = [
                        grid_id
                        for grid_id, fail_count in self.grid_fail_counter.items()
                        if fail_count > self.max_fail_trials_per_grid
                    ]

            unvisited_grids = (
                set(range(self._n_min_trials))
                - set(visited_grids)
                - set(running_grids)
                - set(definitively_failed_grids)
            )
            loop = True

        # If evaluations for all grids have been started, return grids that have not yet finished
        # because all grids should be evaluated before stopping the optimization.
        # if len(unvisited_grids) == 0:
        #     unvisited_grids = set(range(self._n_min_trials)) - set(visited_grids) - set(definitively_failed_grids)
        self.last_time = time.time()
        return list(unvisited_grids)

    def _same_search_space(
        self, search_space: Mapping[str, Sequence[GridValueType]]
    ) -> bool:
        if set(search_space.keys()) != set(self._search_space.keys()):
            return False

        for param_name in search_space.keys():
            if len(search_space[param_name]) != len(self._search_space[param_name]):
                return False

            param_values = cast(
                SortableParamValueSequenceType, search_space[param_name]
            )
            for i, param_value in enumerate(sorted(param_values)):
                if param_value != self._search_space[param_name][i]:
                    return False

        return True

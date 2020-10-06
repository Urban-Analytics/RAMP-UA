import numpy as np

from ramp.summary import Summary
from ramp.snapshot import Snapshot


def test_summary_update():
    npeople = 50 + 34 + 101 + 551
    summary = Summary(snapshot=Snapshot.random(nplaces=10, npeople=npeople, nslots=10), max_time=20)

    time = 10

    statuses = np.concatenate((
        np.full(50, 0),
        np.full(34, 1),
        np.full(101, 4),
        np.full(551, 6),
    ))
    np.random.shuffle(statuses)

    summary.update(time, statuses)

    assert summary.total_counts[0][time] == 50
    assert summary.total_counts[1][time] == 34
    assert summary.total_counts[2][time] == 0
    assert summary.total_counts[3][time] == 0
    assert summary.total_counts[4][time] == 101
    assert summary.total_counts[5][time] == 0
    assert summary.total_counts[6][time] == 551

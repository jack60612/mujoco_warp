# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests the deterministic scatter pool of the actuator-moment kernels."""

from absl.testing import absltest

import mujoco_warp as mjw
from mujoco_warp import test_data

from . import derivative


class ActuatorMomentRecordsTest(absltest.TestCase):
  def test_pool_covers_every_moment_nonzero_and_is_smaller_than_nv_per_actuator(self):
    _, _, m, d = test_data.fixture("humanoid/humanoid.xml", nworld=2)
    mjw.forward(m, d)
    records = derivative.actuator_moment_records(m, d)

    # A launch scatters at most one record per moment nonzero, pooled over its threads: the pool
    # holds nJmom records per world.
    self.assertGreaterEqual(m.nactuator * records, d.actuator_moment.shape[1])
    self.assertGreaterEqual(m.nactuator * records, int(d.moment_rownnz.numpy().sum(axis=1).max()))
    # The previous reservation was nv records for each actuator.
    self.assertLess(records, m.nv)


if __name__ == "__main__":
  absltest.main()

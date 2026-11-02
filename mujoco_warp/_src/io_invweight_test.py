# Copyright 2026 The Newton Developers
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

"""Regression tests for body inverse weights recomputed by set_const."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjwarp
from mujoco_warp import test_data

_XML = """
<mujoco>
  <option jacobian="{jacobian}"/>
  <worldbody>
    <body name="hinge">
      <joint type="hinge"/>
      <inertial pos="0 0 0" mass="2" diaginertia="1 2 3"/>
    </body>
    <body name="ball">
      <joint type="ball"/>
      <inertial pos="0 0 0" mass="2" diaginertia="1 2 3"/>
    </body>
    <body name="slide">
      <joint type="slide" axis="1 0 0"/>
      <joint type="slide" axis="0 1 0"/>
      <joint type="slide" axis="0 0 1"/>
      <inertial pos="0 0 0" mass="2" diaginertia="1 2 3"/>
    </body>
    <body name="free">
      <freejoint/>
      <inertial pos="0 0 0" mass="2" diaginertia="1 2 3"/>
    </body>
    <body name="static">
      <inertial pos="0 0 0" mass="2" diaginertia="1 2 3"/>
    </body>
  </worldbody>
</mujoco>
"""


class BodyInvweightTest(parameterized.TestCase):
  """Checks translational and rotational components independently against MuJoCo."""

  @parameterized.named_parameters(("dense", "dense"), ("sparse", "sparse"))
  def test_set_const_body_invweight0(self, jacobian):
    """Zero mobility in one component must not inherit the other component."""
    mjm, mjd, m, d = test_data.fixture(xml=_XML.format(jacobian=jacobian), nworld=2)
    expected = np.array([[0, 0], [0, 1 / 9], [0, 11 / 18], [1 / 2, 0], [1 / 2, 11 / 18], [0, 0]])
    np.testing.assert_allclose(mjm.body_invweight0, expected, rtol=1e-6, atol=0)

    # Give each world different mass/inertia and force set_const to overwrite stale weights.
    scales = np.array([1.0, 2.0], dtype=np.float32)
    body_mass = scales[:, None] * mjm.body_mass[None, :]
    body_inertia = scales[:, None, None] * mjm.body_inertia[None, :, :]
    m.body_mass = wp.array(body_mass, dtype=float)
    m.body_inertia = wp.array(body_inertia, dtype=wp.vec3)
    m.body_invweight0 = wp.full((2, mjm.nbody), value=wp.vec2(-1.0, -1.0), dtype=wp.vec2)

    mjwarp.set_const(m, d)

    actual = m.body_invweight0.numpy()
    for world in range(2):
      mjm.body_mass[:] = body_mass[world]
      mjm.body_inertia[:] = body_inertia[world]
      mujoco.mj_setConst(mjm, mjd)
      np.testing.assert_allclose(actual[world], mjm.body_invweight0, rtol=1e-5, atol=0)
      np.testing.assert_allclose(actual[world], expected / scales[world], rtol=1e-5, atol=0)


if __name__ == "__main__":
  absltest.main()

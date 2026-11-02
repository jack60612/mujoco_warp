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

"""Sparse passive tendon damping derivative regressions."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import derivative


_XML = """
<mujoco>
  <option integrator="implicitfast" timestep="0.003" gravity="0 0 0"/>
  <default><geom type="sphere" size=".05" contype="0" conaffinity="0"/></default>
  <worldbody>
    <body>
      <joint name="root" type="slide" axis="1 0 0"/>
      <geom/>
      <site name="anchor" pos="0 0 .2"/>
      <body pos="0 0 1">
        <joint name="left" type="slide" axis="0 0 1"/>
        <geom/><site name="left_site"/>
      </body>
      <body pos="1 0 1">
        <joint name="right" type="slide" axis="0 0 1"/>
        <geom/><site name="right_site"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed damping="2 .3 .2">
      <joint joint="root" coef="1"/>
      <joint joint="left" coef="-2"/>
      <joint joint="right" coef=".5"/>
    </fixed>
    <fixed damping="0 .4 .1">
      <joint joint="left" coef="1"/>
    </fixed>
    <spatial damping="3">
      <site site="anchor"/><site site="left_site"/><site site="right_site"/>
    </spatial>
    <fixed damping="0"><joint joint="root" coef="1"/></fixed>
  </tendon>
  <keyframe><key qvel=".2 -.7 .4"/></keyframe>
</mujoco>
"""


class TendonDampingTest(parameterized.TestCase):
  @parameterized.product(jacobian=(0, 1), disabled=(False, True))
  def test_native_derivative(self, jacobian, disabled):
    """Match native implicitfast, including polynomial and off-pattern entries."""
    flags = int(mujoco.mjtDisableBit.mjDSBL_DAMPER) if disabled else 0
    mjm, mjd, m, d = test_data.fixture(
      xml=_XML,
      keyframe=0,
      nworld=3,
      overrides={"opt.jacobian": jacobian, "opt.disableflags": flags},
    )
    out = wp.zeros((d.nworld, m.nC), dtype=float)
    mjw.deriv_smooth_vel(m, d, out)
    mujoco.mj_step(mjm, mjd)
    native = np.zeros((m.nv, m.nv))
    mujoco.mju_sparse2dense(native, mjd.qDeriv, mjm.D_rownnz, mjm.D_rowadr, mjm.D_colind)
    mass = np.zeros_like(native)
    mujoco.mju_sym2dense(mass, mjd.M, mjm.M_rownnz, mjm.M_rowadr, mjm.M_colind)
    expected = mass - mjm.opt.timestep * native
    # MJWarp implicitfast preserves M sparsity, not all of native D's entries.
    for world in out.numpy():
      for i in range(m.nv):
        adr = mjm.M_rowadr[i]
        cols = mjm.M_colind[adr : adr + mjm.M_rownnz[i]]
        np.testing.assert_allclose(world[adr : adr + len(cols)], expected[i, cols], atol=2e-6, rtol=2e-6)

  def test_world_parameters_and_zero_jacobian(self):
    """Scatter overlapping tendons with broadcast parameters and missing M pairs."""
    # Lower M pattern: (0,0), (1,0), (1,1), (2,0), (2,2).
    elemid = np.array([[0, 1, 3], [1, 2, -1], [3, -1, 4]], dtype=np.int32)
    jac = np.array([[1, -2, .5, 0, 3], [-1, 0, 2, 4, 0]], dtype=np.float32)
    velocity = np.array([[-2, 1], [3, -1]], dtype=np.float32)
    damping = np.array([[2, 0], [0, 1]], dtype=np.float32)
    poly = np.array([[[.3, .2], [.4, .1]]], dtype=np.float32)
    timestep = np.array([.01, .02], dtype=np.float32)
    out = wp.full((2, 5), 7.0, dtype=float)
    wp.launch(
      derivative._qderiv_tendon_damping,
      dim=(2, 2),
      inputs=[
        wp.array(timestep),
        wp.array(np.array([3, 2], dtype=np.int32)),
        wp.array(np.array([0, 3], dtype=np.int32)),
        wp.array(np.array([0, 1, 2, 0, 1], dtype=np.int32)),
        wp.array(damping),
        wp.array(poly, dtype=wp.vec2),
        wp.array(elemid),
        wp.array(jac),
        wp.array(velocity),
      ],
      outputs=[out],
    )
    expected = np.full((2, 5), 7.0)
    for w in range(2):
      for t, (start, stop, cols) in enumerate(((0, 3, [0, 1, 2]), (3, 5, [0, 1]))):
        row = jac[w, start:stop]
        v = abs(velocity[w, t])
        deriv = damping[w, t] + 2 * poly[0, t, 0] * v + 3 * poly[0, t, 1] * v * v
        for i, ci in enumerate(cols):
          for j, cj in enumerate(cols[: i + 1]):
            if elemid[ci, cj] >= 0:
              expected[w, elemid[ci, cj]] += timestep[w] * row[i] * row[j] * deriv
    np.testing.assert_allclose(out.numpy(), expected, atol=2e-6, rtol=2e-6)

  def test_implicitfast_steps(self):
    """Keep native passive damping dynamics on a serial chain."""
    xml = """
      <mujoco>
        <option integrator="implicitfast" gravity="0 0 0"/>
        <default><geom size=".1" contype="0" conaffinity="0"/></default>
        <worldbody>
          <body>
            <joint name="a" type="slide" axis="1 0 0"/><geom/>
            <body pos="0 0 1">
              <joint name="b" type="slide" axis="0 0 1"/><geom/>
            </body>
          </body>
        </worldbody>
        <tendon>
          <fixed damping="2 .3 .2">
            <joint joint="a" coef="1"/><joint joint="b" coef="-2"/>
          </fixed>
        </tendon>
        <keyframe><key qvel=".2 -.7"/></keyframe>
      </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, keyframe=0)
    for _ in range(20):
      mujoco.mj_step(mjm, mjd)
      mjw.step(m, d)
    np.testing.assert_allclose(d.qpos.numpy()[0], mjd.qpos, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(d.qvel.numpy()[0], mjd.qvel, atol=2e-6, rtol=2e-6)



if __name__ == "__main__":
  absltest.main()

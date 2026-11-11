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

"""Opt-in implicitfast damping couplings outside the kinematic-tree sparsity of M."""

import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import derivative
from mujoco_warp._src import forward
from mujoco_warp._src import smooth

# Two kinematic trees: a hinge parent with two sibling branches, and a six-dof child (three slides
# and a ball, kinematically a free joint but not a standalone free body, whose rows implicitfast
# replaces by a local block) held to it by a CONNECT. Site/refsite actuators and a spatial tendon
# damp the child against the parent (cross-tree pairs); a fixed tendon damps the two siblings (same
# tree, but outside M's ancestor sparsity).
_TWO_TREE_XML = """
<mujoco>
  <option integrator="implicitfast" timestep="0.002"/>
  <default><geom type="capsule" size=".02" contype="0" conaffinity="0"/></default>
  <worldbody>
    <body name="parent">
      <joint name="parent_x" type="hinge" axis="1 0 0"/>
      <joint name="parent_y" type="hinge" axis="0 1 0"/>
      <geom fromto="0 0 0 0 0 .4"/>
      <site name="parent_tip" pos="0 0 .4"/>
      <site name="parent_side" pos=".05 0 .35"/>
      <body name="left" pos="0 0 .2">
        <joint name="left" type="hinge" axis="0 1 0"/>
        <geom fromto="0 0 0 .2 0 .1"/>
      </body>
      <body name="right" pos="0 0 .2">
        <joint name="right" type="hinge" axis="1 0 0"/>
        <geom fromto="0 0 0 -.2 0 .1"/>
      </body>
    </body>
    <body name="child" pos="0 0 .4">
      <joint type="slide" axis="1 0 0"/>
      <joint type="slide" axis="0 1 0"/>
      <joint type="slide" axis="0 0 1"/>
      <joint type="ball"/>
      <geom fromto="0 0 0 .05 0 .3"/>
      <site name="child_base"/>
      <site name="child_side" pos=".08 .02 .1"/>
    </body>
  </worldbody>
  <equality>
    <connect body1="child" body2="parent" anchor="0 0 0"/>
  </equality>
  <tendon>
    <fixed name="siblings" damping="3">
      <joint joint="left" coef="1"/>
      <joint joint="right" coef="-.5"/>
    </fixed>
    <spatial name="trees" damping="4">
      <site site="parent_side"/>
      <site site="child_side"/>
    </spatial>
  </tendon>
  <actuator>
    <general site="child_base" refsite="parent_tip" gear="0 0 0 1 0 0" biastype="affine" biasprm="0 -5 -2"/>
    <general site="child_base" refsite="parent_tip" gear="0 0 0 0 1 0" biastype="affine" biasprm="0 -5 -3"/>
    <general joint="left" biastype="affine" biasprm="0 0 -1"/>
  </actuator>
  <keyframe>
    <key qpos=".1 -.2 .3 -.4 .01 -.02 .01 .995 .05 -.05 .07"
         qvel=".5 -.3 .8 -.6 .2 -.1 .3 1.5 -2 .7"/>
  </keyframe>
</mujoco>
"""

# The two-tree model with a FREE child, i.e. a standalone free body: its site/refsite actuators
# damp it against the parent tree (as declared break points do in farm-sim). implicitfast solves
# its rows with the local free-body block, which must keep the actuators' own damping. The child
# starts without angular velocity, so the block's gyroscopic derivative is zero.
_FREE_TWO_TREE_XML = _TWO_TREE_XML.replace(
  """<joint type="slide" axis="1 0 0"/>
      <joint type="slide" axis="0 1 0"/>
      <joint type="slide" axis="0 0 1"/>
      <joint type="ball"/>""",
  "<freejoint/>",
).replace(
  """<key qpos=".1 -.2 .3 -.4 .01 -.02 .01 .995 .05 -.05 .07"
         qvel=".5 -.3 .8 -.6 .2 -.1 .3 1.5 -2 .7"/>""",
  """<key qpos=".1 -.2 .3 -.4 .01 -.02 .41 .995 .05 -.05 .07"
         qvel=".5 -.3 .8 -.6 .2 -.1 .3 0 0 0"/>""",
)

# The two-tree model with a damped standalone free child, coupled to nothing: only the siblings'
# tendon is outside M's sparsity. implicitfast solves the free body with its local block.
_FREE_CHILD_XML = """
<mujoco>
  <option integrator="implicitfast" timestep="0.002"/>
  <default><geom type="capsule" size=".02" contype="0" conaffinity="0"/></default>
  <worldbody>
    <body name="parent">
      <joint name="parent_x" type="hinge" axis="1 0 0"/>
      <geom fromto="0 0 0 0 0 .2"/>
      <body name="left" pos="0 0 .2">
        <joint name="left" type="hinge" axis="0 1 0"/>
        <geom fromto="0 0 0 .2 0 .1"/>
      </body>
      <body name="right" pos="0 0 .2">
        <joint name="right" type="hinge" axis="1 0 0"/>
        <geom fromto="0 0 0 -.2 0 .1"/>
      </body>
    </body>
    <body name="child" pos="0 0 .4">
      <joint type="free" damping=".5"/>
      <geom fromto="0 0 0 .05 0 .3"/>
    </body>
  </worldbody>
  <tendon>
    <fixed name="siblings" damping="3">
      <joint joint="left" coef="1"/>
      <joint joint="right" coef="-.5"/>
    </fixed>
  </tendon>
  <keyframe>
    <key qvel=".5 -.3 .8 .2 -.1 .3 1.5 -2 .7"/>
  </keyframe>
</mujoco>
"""

# Three slide bodies in three trees, coupled by one stiffly damped tendon actuator. The native
# tree-block solve treats the cross-tree part explicitly and diverges at this timestep.
_THREE_TREE_XML = """
<mujoco>
  <option integrator="implicitfast" timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    <body><joint name="a" type="slide" axis="1 0 0"/><geom size=".1" mass="1" contype="0" conaffinity="0"/></body>
    <body pos="0 1 0"><joint name="b" type="slide" axis="1 0 0"/><geom size=".1" mass="1" contype="0" conaffinity="0"/></body>
    <body pos="0 2 0"><joint name="c" type="slide" axis="1 0 0"/><geom size=".1" mass="1" contype="0" conaffinity="0"/></body>
  </worldbody>
  <tendon>
    <fixed name="sum"><joint joint="a" coef="1"/><joint joint="b" coef="1"/><joint joint="c" coef="1"/></fixed>
  </tendon>
  <actuator>
    <general tendon="sum" biastype="affine" biasprm="0 0 -1000"/>
  </actuator>
  <keyframe><key qvel="1 -.5 .2"/></keyframe>
</mujoco>
"""


def _lower_mask(m):
  elemid = m.M_elemid.numpy()
  return elemid, (elemid >= 0) | (elemid.T >= 0)


def _dense_m(m, d, world=0):
  elemid, _ = _lower_mask(m)
  M = d.M.numpy()[world].astype(np.float64)
  dense = np.zeros((m.nv, m.nv))
  for i in range(m.nv):
    for j in range(i + 1):
      if elemid[i, j] >= 0:
        dense[i, j] = dense[j, i] = M[elemid[i, j]]
  return dense


def _damping_hessian(mjm, m, d, world=0):
  """Returns dt * B, B = -dforce/dqvel of actuator bias and passive damping (float64, dense)."""
  dt = mjm.opt.timestep
  B = np.diag(mjm.dof_damping.astype(np.float64))

  rownnz = d.moment_rownnz.numpy()[world]
  rowadr = d.moment_rowadr.numpy()[world]
  colind = d.moment_colind.numpy()[world]
  moment = d.actuator_moment.numpy()[world].astype(np.float64)
  for a in range(mjm.nu):
    row = np.zeros(m.nv)
    for k in range(rowadr[a], rowadr[a] + rownnz[a]):
      row[colind[k]] += moment[k]
    B -= mjm.actuator_biasprm[a, 2] * np.outer(row, row)

  ten_J = d.ten_J.numpy()[world].astype(np.float64)
  for t in range(mjm.ntendon):
    row = np.zeros(m.nv)
    for k in range(mjm.ten_J_rowadr[t], mjm.ten_J_rowadr[t] + mjm.ten_J_rownnz[t]):
      row[mjm.ten_J_colind[k]] += ten_J[k]
    B += mjm.tendon_damping[t] * np.outer(row, row)

  return dt * B


def _reference_qvel(mjm, m, d, pattern_only, world=0):
  """Returns qvel + dt * (M + dt B)^-1 Ma in float64, optionally restricted to M's sparsity."""
  A = _dense_m(m, d, world) + _damping_hessian(mjm, m, d, world)
  if pattern_only:
    A = np.where(_lower_mask(m)[1], A, 0.0)
  Ma = d.efc.Ma.numpy()[world].astype(np.float64)
  qvel = d.qvel.numpy()[world].astype(np.float64)
  return qvel + mjm.opt.timestep * np.linalg.solve(A, Ma)


class ImplicitCrossTreeTest(parameterized.TestCase):
  def test_default_disabled(self):
    """The option is off unless requested."""
    _, _, m, _ = test_data.fixture(xml=_TWO_TREE_XML, keyframe=0)
    self.assertEqual(m.opt.implicit_crosstree_iterations, 0)

  def test_mul_matches_dense_omitted_pairs(self):
    """The matrix-free product applies exactly the pairs deriv_smooth_vel leaves out."""
    mjm, _, m, d = test_data.fixture(xml=_TWO_TREE_XML, keyframe=0, nworld=2)
    mjw.forward(m, d)
    out = wp.empty((d.nworld, m.nC), dtype=float)
    vel = wp.empty((d.nworld, m.nactuator), dtype=float)
    mjw.deriv_smooth_vel(m, d, out, vel)
    np.testing.assert_allclose(vel.numpy(), np.tile(mjm.actuator_biasprm[:, 2], (2, 1)))

    vec = np.random.default_rng(0).uniform(-1, 1, (d.nworld, m.nv)).astype(np.float32)
    res = wp.zeros((d.nworld, m.nv), dtype=float)
    derivative.crosstree_damping_mul(m, d, vel, wp.array(vec, dtype=float), res)

    _, in_pattern = _lower_mask(m)
    for world in range(d.nworld):
      omitted = np.where(in_pattern, 0.0, _damping_hessian(mjm, m, d, world))
      self.assertGreater(np.abs(omitted).max(), 1e-3)  # the model really has omitted pairs
      np.testing.assert_allclose(res.numpy()[world], omitted @ vec[world], rtol=1e-5, atol=1e-5)

      # the in-pattern part is exactly what deriv_smooth_vel stores
      stored = np.where(in_pattern, _dense_m(m, d, world) + _damping_hessian(mjm, m, d, world), 0.0)
      elemid, _ = _lower_mask(m)
      for i in range(m.nv):
        for j in range(i + 1):
          if elemid[i, j] >= 0:
            np.testing.assert_allclose(out.numpy()[world, elemid[i, j]], stored[i, j], rtol=1e-5, atol=1e-5)

  def test_two_tree_matches_dense_implicit_reference(self):
    """With enough iterations the step equals the dense float64 solve including omitted pairs."""
    mjm, _, m, d = test_data.fixture(xml=_TWO_TREE_XML, keyframe=0)
    mjw.forward(m, d)
    full = _reference_qvel(mjm, m, d, pattern_only=False)
    native = _reference_qvel(mjm, m, d, pattern_only=True)
    self.assertGreater(np.abs(full - native).max(), 1e-3)  # the omitted pairs matter

    m.opt.implicit_crosstree_iterations = m.nv
    mjw.implicit(m, d)
    np.testing.assert_allclose(d.qvel.numpy()[0], full, rtol=1e-4, atol=1e-5)

  def test_two_tree_disabled_matches_native_reference(self):
    """Disabled, the step keeps the native tree-sparse solve."""
    mjm, _, m, d = test_data.fixture(xml=_TWO_TREE_XML, keyframe=0)
    mjw.forward(m, d)
    native = _reference_qvel(mjm, m, d, pattern_only=True)
    mjw.implicit(m, d)
    np.testing.assert_allclose(d.qvel.numpy()[0], native, rtol=1e-4, atol=1e-5)

  @parameterized.parameters(0, 1)
  def test_free_child_matches_dense_reference(self, enabled):
    """A standalone free child keeps its own actuator damping implicit in its local block.

    Disabled, the step equals the dense solve restricted to M's sparsity, which includes the free
    body's diagonal block; enabled with nv iterations, it also includes the cross-tree pairs.
    """
    mjm, _, m, d = test_data.fixture(xml=_FREE_TWO_TREE_XML, keyframe=0)
    self.assertEqual(m.body_freeadr.size, 1)
    mjw.forward(m, d)
    reference = _reference_qvel(mjm, m, d, pattern_only=not enabled)
    m.opt.implicit_crosstree_iterations = m.nv if enabled else 0
    mjw.implicit(m, d)
    np.testing.assert_allclose(d.qvel.numpy()[0], reference, rtol=1e-4, atol=1e-5)

  def test_disabled_is_the_stock_pipeline(self):
    """Disabled, implicitfast is bitwise the factor-solve of deriv_smooth_vel's matrix."""
    _, _, m, d = test_data.fixture(xml=_TWO_TREE_XML, keyframe=0)
    _, _, m_ref, d_ref = test_data.fixture(xml=_TWO_TREE_XML, keyframe=0)
    mjw.forward(m, d)
    mjw.forward(m_ref, d_ref)

    mjw.implicit(m, d)

    qDeriv = wp.empty((d_ref.nworld, m_ref.nC), dtype=float)
    qLD = wp.empty_like(d_ref.qLD)
    qLDiagInv = wp.empty((d_ref.nworld, m_ref.nv), dtype=float)
    qacc = wp.empty((d_ref.nworld, m_ref.nv), dtype=float)
    mjw.deriv_smooth_vel(m_ref, d_ref, qDeriv)
    smooth.factor_solve_i(m_ref, d_ref, qDeriv, qLD, qLDiagInv, qacc, d_ref.efc.Ma)
    forward._advance(m_ref, d_ref, qacc)

    np.testing.assert_array_equal(d.qvel.numpy(), d_ref.qvel.numpy())
    np.testing.assert_array_equal(d.qpos.numpy(), d_ref.qpos.numpy())

  def test_two_iterations_reduce_error(self):
    """Each fixed PCG iteration reduces the energy-norm error of the dense reference solve."""
    errors = []
    for iterations in (0, 1, 2):
      mjm, _, m, d = test_data.fixture(xml=_TWO_TREE_XML, keyframe=0)
      mjw.forward(m, d)
      full = _reference_qvel(mjm, m, d, pattern_only=False)
      A = _dense_m(m, d) + _damping_hessian(mjm, m, d)
      m.opt.implicit_crosstree_iterations = iterations
      mjw.implicit(m, d)
      e = d.qvel.numpy()[0] - full
      errors.append(np.sqrt(e @ A @ e))
    self.assertLess(errors[1], errors[0])
    self.assertLess(errors[2], errors[1])

  def test_free_body_rows_keep_their_local_solve(self):
    """Free-body rows enter the PCG with their local block, so the refinement leaves them."""
    _, _, m, d = test_data.fixture(xml=_FREE_CHILD_XML, keyframe=0)
    _, _, m_ref, d_ref = test_data.fixture(xml=_FREE_CHILD_XML, keyframe=0)
    self.assertEqual(m.body_freeadr.size, 1)
    mjw.forward(m, d)
    mjw.forward(m_ref, d_ref)
    m.opt.implicit_crosstree_iterations = 3
    mjw.implicit(m, d)
    mjw.implicit(m_ref, d_ref)

    qvel, qvel_ref = d.qvel.numpy()[0], d_ref.qvel.numpy()[0]
    self.assertGreater(np.abs(qvel[:3] - qvel_ref[:3]).max(), 1e-5)  # the siblings' coupling is added
    np.testing.assert_allclose(qvel[3:], qvel_ref[3:], rtol=1e-6, atol=1e-7)

  @parameterized.parameters(0, 2)
  def test_three_tree_damper_stability(self, iterations):
    """A damper across three trees diverges natively and stays stable when enabled."""
    _, _, m, d = test_data.fixture(xml=_THREE_TREE_XML, keyframe=0)
    m.opt.implicit_crosstree_iterations = iterations
    for _ in range(100):
      mjw.step(m, d)
    qvel = d.qvel.numpy()[0]
    if iterations:
      # (1, 1, 1) mode decays; the two orthogonal modes keep their momentum exactly
      self.assertTrue(np.all(np.isfinite(qvel)))
      self.assertLess(abs(qvel.sum()), 1e-3)
      np.testing.assert_allclose(qvel - qvel.mean(), np.array([1, -0.5, 0.2]) - 0.7 / 3, atol=1e-4)
    else:
      self.assertFalse(np.all(np.abs(qvel) < 1e3))

  def test_implicit_integrator_rejected(self):
    _, _, m, d = test_data.fixture(xml=_TWO_TREE_XML, keyframe=0, overrides={"opt.integrator": "IMPLICIT"})
    m.opt.implicit_crosstree_iterations = 2
    with self.assertRaises(NotImplementedError):
      mjw.implicit(m, d)

  def test_override(self):
    _, _, m, _ = test_data.fixture(xml=_TWO_TREE_XML, overrides={"opt.implicit_crosstree_iterations": "3"})
    self.assertEqual(m.opt.implicit_crosstree_iterations, 3)

  def test_graph_capture(self):
    """The fixed iteration count captures into a CUDA graph and replays like eager steps."""
    if not wp.get_device().is_cuda:
      self.skipTest("CUDA graph capture requires a CUDA device")

    _, _, m, d = test_data.fixture(xml=_TWO_TREE_XML, keyframe=0, nworld=4)
    _, _, m_eager, d_eager = test_data.fixture(xml=_TWO_TREE_XML, keyframe=0, nworld=4)
    m.opt.implicit_crosstree_iterations = 2
    m_eager.opt.implicit_crosstree_iterations = 2

    mjw.step(m, d)  # compile outside capture
    mjw.step(m_eager, d_eager)
    with wp.ScopedCapture() as capture:  # records without executing
      mjw.step(m, d)
    for _ in range(5):
      wp.capture_launch(capture.graph)
      mjw.step(m_eager, d_eager)

    np.testing.assert_allclose(d.qvel.numpy(), d_eager.qvel.numpy(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(d.qpos.numpy(), d_eager.qpos.numpy(), rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
  wp.init()
  absltest.main()

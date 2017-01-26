# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for the cosine distance class.
"""

import numpy as np
import pytest
from .. import *
from ...axis import Axis
from ... import sequence

def test_cosine_distance():
  a = np.reshape(np.arange(25.0, dtype = np.float32), (5,5))
  b = np.reshape(np.arange(0, 5, dtype=np.float32), (1,5))
  
  src = input_variable(shape=(5), dynamic_axes=[ Axis.default_batch_axis(), Axis("Seq")])
  tgt = input_variable(shape=(5))
  tgt_br = sequence.broadcast_as(tgt, src)
  cos_seq = cosine_distance(src, tgt_br)
  assert len(cos_seq.dynamic_axes)==2
  assert cos_seq.dynamic_axes[1].name=="Seq"
  val = cos_seq.eval({src:[a], tgt:[b]})
  print("Cosine similarity\r\n{0}\r\n  #\r\n{1}".format(a,b))
  print("==>")
  print(val)

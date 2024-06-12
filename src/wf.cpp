/*
Author: Humberto Munoz Bauza (humberto.munozbauza@nasa.gov)

Copyright © 2023, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The PySA, a powerful tool for solving optimization problems is licensed under
the Apache License, Version 2.0 (the "License"); you may not use this file
except in compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
*/

#include "sternx/wf.h"
#include <iostream>

std::vector<double> combo_dist(long n, long k, long t) {
  // Evaluate the log-probabilities that, in a random IR split
  // of an [n, k] code, w errors will be contained in the redundancy set,
  // where  0 <= w <= t
  std::vector<double> pdist(t + 1, 0.0);
  double z = log2_binomial(n, k);
  for (long w = 0; w <= t; ++w) {
    if (w == 0) {
      pdist[w] = log2_binomial(n - t, k) - z;
    } else if (w == t) {
      pdist[w] = log2_binomial(n - t, k - t) - z;
    } else {
      pdist[w] = log2_binomial(n - t, k - w) + log2_binomial(t, w) - z;
    }
  }
  return pdist;
}

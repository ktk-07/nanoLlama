from __future__ import annotations

import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp

mesh = jax.make_mesh((4, 2), ('X', 'Y'))
jax.set_mesh(mesh)

x = jnp.arange(8 * 4, dtype=jnp.float32).reshape(8, 4)
x = jax.device_put(x, jax.P(('X', 'Y'),))

print(jax.devices())
print(jax.typeof(x))  # float32[8@X,4@Y]
print(x.sharding)


for s in x.addressable_shards:
    print(s.device, s.data, s.data.shape, sep='\n', end='\n\n')

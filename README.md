luau's math.random uses the pcg32 XSH-RR implementation for generating random numbers. clover attacks it.

the impl for luau's pcg32 can be seen here:
```cpp
#undef PI
#define PI (3.14159265358979323846)
#define RADIANS_PER_DEGREE (PI / 180.0)


#define PCG32_INC 105


static uint32_t pcg32_random(uint64_t* state)
{
    uint64_t oldstate = *state;
    *state = oldstate * 6364136223846793005ULL + (PCG32_INC | 1);
    uint32_t xorshifted = uint32_t(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = uint32_t(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-int32_t(rot)) & 31));
}


static void pcg32_seed(uint64_t* state, uint64_t seed)
{
    *state = 0;
    pcg32_random(state);
    *state += seed;
    pcg32_random(state);
}
....


static int math_random(lua_State* L)
{
    global_State* g = L->global;
    switch (lua_gettop(L))
    { // check number of arguments
    case 0:
    { // no arguments
        // Using ldexp instead of division for speed & clarity.
        // See http://mumble.net/~campbell/tmp/random_real.c for details on generating doubles from integer ranges.
        uint32_t rl = pcg32_random(&g->rngstate);
        uint32_t rh = pcg32_random(&g->rngstate);
        double rd = ldexp(double(rl | (uint64_t(rh) << 32)), -64);
        lua_pushnumber(L, rd); // number between 0 and 1
        break;
    }
    case 1:
    { // only upper limit
        int u = luaL_checkinteger(L, 1);
        luaL_argcheck(L, 1 <= u, 1, "interval is empty");

        uint64_t x = uint64_t(u) * pcg32_random(&g->rngstate);
        int r = int(1 + (x >> 32));
        lua_pushinteger(L, r); // int between 1 and `u'
        break;
    }
    case 2:
    { // lower and upper limits
        int l = luaL_checkinteger(L, 1);
        int u = luaL_checkinteger(L, 2);
        luaL_argcheck(L, l <= u, 2, "interval is empty");

        uint32_t ul = uint32_t(u) - uint32_t(l);
        luaL_argcheck(L, ul < UINT_MAX, 2, "interval is too large"); // -INT_MIN..INT_MAX interval can result in integer overflow
        uint64_t x = uint64_t(ul + 1) * pcg32_random(&g->rngstate);
        int r = int(l + (x >> 32));
        lua_pushinteger(L, r); // int between `l' and `u'
        break;
    }
    default:
        luaL_error(L, "wrong number of arguments");
    }
    return 1;
}
```

### cracking pcg32
to do this, clover enumerates all 32 possible rotations from 0 to 31 rotations inclusive. then, for each rotation, reconstruct the 'high bits' (59..27) of the pre-rotation `xorshifted` value (see above). these become the high bits of the (old state >> 18)  XOR old state. assuming this goes fine, we bruteforce the lower 27 bits lost in the 27 bitshift through GPU (cuda magic). then, we 'stamp' the 5-bit rot in the state, as `rot = old >> 59`; which implies that this participates in the later permutation stage of pcg. this must match the rotation we are testing. so we set `FULL_XOR |= (rot << 59)` to make it actually consistent (from olds 59-63 bits). finally, we unxorshift per and from the above to recover our old state.

the pcg32 xsh-rr algorithm is slightly harder to break than lcgs (duh) but it is still relatively easy if you have a good rig.



Rejected startup, not a benchmark. Exit 0xc0000005 before model loading.

The cached Ninja build had zero header dependencies on old C++ objects; the changed Engine interface was
linked with stale architecture objects. Removing all engine C++ objects and rebuilding fixed startup.
The later build records engine.hpp dependencies again. No performance result is taken from this directory.

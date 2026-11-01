# AGENTS.md

PolyStrata runs the Qwen3.8-Flash-Next mixture-of-experts model (and its Coder, Swift 1.5 and Unsloth variants) on a
normal PC: one NVIDIA or AMD graphics card plus system RAM, on Windows or Linux. It has a C++/CUDA/HIP engine
(`src/`, `include/`), a Python server with an OpenAI- and Anthropic-compatible API and a web app (`serve/`), and a
one-click installer (`setup.py`, started by `START-HERE.bat` / `setup.sh`).

A second engine, `strata-poly` (`src/poly/`), runs GLM 5.3 Flash, Ornith 1.5 (35B and 122B), DeepSeek V4 Flash and other GGUF models on Linux or Windows: [docs/GLM.md](docs/GLM.md), [docs/ORNITH.md](docs/ORNITH.md), [docs/DEEPSEEK.md](docs/DEEPSEEK.md), [docs/BACKENDS.md](docs/BACKENDS.md). The Qwen engine, the installer, the server and the web app come from
[Strata](https://github.com/Niko1221/Strata).

## Installing PolyStrata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the model by RAM, run setup non-interactively,
start and verify the server, and connect the user's apps. Never expose the server beyond `127.0.0.1` without
`--api-key`. As an alternative to shell commands, PolyStrata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md))
offers the same steps as tools.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- Text taken from the Strata project names that project. After merging from it,
  `python tools/polystrata_names.py` writes PolyStrata again; its first lines say what it leaves alone.
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py` (for example
  `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.

## Running a command that takes a while

Before starting a command that blocks the session and leaves the user without an answer for more than a few
seconds — a model load, a benchmark, a build, a download — write in the chat what is started and how long the
reply will take (for example "starting the server, about 2 minutes"). Only then run it. A command that takes
minutes is started with the wait said out loud, never silently.

- Every long command is announced before it starts, with what it is and how long it will take. A command
  that is not announced is not started.
- A wait between checks is at most 120 seconds (`Start-Sleep -Seconds 120` at most). A longer sleep wastes
  the user's time: the answer arrives when it is needed, not when the sleep ends. Wake up sooner, look, and
  if the command is still running, sleep again.
- The wait is never the answer: after every wait, report what the run has produced so far, even if it is
  only that the run is still going.

## Never destroy work in progress

- A running command, experiment or test is never stopped, killed, restarted or rerun because a message
  arrived. A long command is interrupted only when that message explicitly says to stop, cancel, kill or
  delete it.
- A message that only points out a preference — the wrong disk, the wrong flag, a better way — does not
  authorize stopping, deleting or restarting anything. It applies to the next run, not to the one already
  in flight or already finished.
- Results are never deleted or overwritten: `bench/results/**`, its logs, `summary.json`, `results.json`,
  status files, raw traces and experiment work directories. A run is redone only when the user asks for
  that exact run to be redone, and the previous result is not deleted to make room without an explicit
  order.
- When a remark could read either as "change the next run" or "redo this one", read it as the former and
  ask before touching anything that has already run.

## Contributing a change or report

- Search the open issues and pull requests first, and add to a thread that already covers your point.
- Open an issue with the form that fits (bug report, feature request or question).
- One change per pull request. Say what it changes and what it leaves alone.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).

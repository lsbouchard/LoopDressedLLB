# Examples

`pra_benchmark_commands.sh` contains the PRA benchmark command suite for the
`jeener_packet_gaussian.py` executable.

The script is meant to be run from a working directory where the command
`jeener-packet-gaussian` is available. It writes figures and logs to
`pra_benchmark_outputs/`.

The script defaults to `MPLBACKEND=Agg` so it can run in batch or CI contexts.

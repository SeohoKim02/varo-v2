"""Local launcher that caps BLAS threads before numpy is imported.

Some constrained machines hit an OpenBLAS allocation failure when the analysis
pipeline (sklearn/scipy) runs inside Streamlit's ScriptRunner thread. Setting the
thread counts before any numpy import avoids that. This is only a run helper; it
does not change app behavior or results.
"""
import os
import sys

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from streamlit.web import cli as stcli  # noqa: E402  (after env setup)

if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "8541"
    sys.argv = [
        "streamlit", "run", "app_v2.py",
        "--server.headless=true",
        f"--server.port={port}",
        "--server.address=127.0.0.1",
    ]
    sys.exit(stcli.main())

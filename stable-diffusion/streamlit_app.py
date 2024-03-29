import shlex
import subprocess
from pathlib import Path

import modal

image = modal.Image.debian_slim().pip_install("streamlit", "numpy", "pandas","requests")

stub = modal.Stub(name="example-modal-streamlit", image=image)

streamlit_script_local_path = Path(__file__).parent / "app.py"
streamlit_script_remote_path = Path("/root/app.py")

if not streamlit_script_local_path.exists():
    raise RuntimeError(
        "app.py not found! Place the script with your streamlit app in the same directory."
    )

streamlit_script_mount = modal.Mount.from_local_file(
    streamlit_script_local_path,
    streamlit_script_remote_path,
)

# ## Spawning the Streamlit server
#
# Inside the container, we will run the Streamlit server in a background subprocess using
# `subprocess.Popen`. We also expose port 8000 using the `@web_server` decorator.
@stub.function(
    allow_concurrent_inputs=100,
    mounts=[streamlit_script_mount],
    container_idle_timeout=240
)
@modal.web_server(8000)
def run():
    target = shlex.quote(str(streamlit_script_remote_path))
    cmd = f"streamlit run {target} --server.port 8000 --server.enableCORS=false --server.enableXsrfProtection=false"
    subprocess.Popen(cmd, shell=True)
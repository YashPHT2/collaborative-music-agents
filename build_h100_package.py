"""Build a portable notebook + audited payload, excluding credentials/raw archives."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import textwrap
import zipfile

ROOT=Path(__file__).resolve().parent
DIST=ROOT/'outputs/h100_training_package_v1'
PAYLOAD=DIST/'h100_music_package'
ARCHIVE_NAME='h100_music_training_v1.zip'
NOTEBOOK_NAME='H100_Music_Training.ipynb'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()


def copy(relative,destination=None):
    source=ROOT/relative;target=PAYLOAD/(destination or relative);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)


def code_closure(seeds):
    pending=list(seeds);seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        path=ROOT/(name+'.py')
        if not path.exists():continue
        seen.add(name)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            if isinstance(node,ast.Import):pending.extend(item.name.split('.')[0] for item in node.names)
            if isinstance(node,ast.ImportFrom) and node.module:pending.append(node.module.split('.')[0])
    return sorted(seen)


def notebook():
    import nbformat as nbf
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))
    def py(s):cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip()))
    md('''
    # H100 / A100: train the orchestra together
    **Upload `h100_music_training_v1.zip` and this notebook to your GPU notebook.**
    H100 80GB and A100 40/80GB runtimes are supported. Batch size is measured on
    the GPU actually assigned to you. This notebook also works with the original
    v1 ZIP: no package re-upload or environment reinstall is needed for this fix.
    Use Python **3.10–3.12**, internet access, and at least **15 GB free disk**.
    The workflow installs an isolated environment, verifies files, benchmarks the
    actual GPU, trains all 19 role slots through 7 shared family adapters, then
    generates matched before/after audio. The 233M backbone remains frozen.

    The package contains 6,944 aligned training phrases, 2,419 validation phrases,
    and 1,254 untouched test phrases. It starts from the laptop's selected phrase
    checkpoint (64 updates). Its partner stage stopped on a duration-boundary
    bug; the package includes the fix and does **not** claim that stage finished.

    **This is supervised training, not RL.** “Calm”, major-like pitch statistics
    and rendering controls are not human mood labels. Better music or concentration
    is an evaluation question. Full-model fine-tuning, FP8, and 8–16 bar training
    targets are deliberately not enabled in this verified first package.
    ''')
    py('''
    from pathlib import Path
    import os, sys, subprocess, zipfile, json, hashlib, shutil, time, signal
    from IPython.display import display, Audio, Image, FileLink, clear_output

    # EDIT these paths for your provider. ZIP_PATH is the archive you upload.
    ZIP_PATH = Path.cwd() / "h100_music_training_v1.zip"
    EXTRACT_ROOT = Path.cwd() / "music_workspace"
    # IMPORTANT: set this to a mounted persistent disk / Drive folder if available.
    RESULTS_ROOT = Path(os.environ.get("MUSIC_RUN_ROOT", str(Path.cwd() / "music_results")))
    PROJECT = EXTRACT_ROOT / "h100_music_package"
    RUN = RESULTS_ROOT / "h100_orchestra_v1"
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError("Select a Python 3.10–3.12 notebook runtime for the pinned environment.")
    if not ZIP_PATH.is_file():
        raise FileNotFoundError(f"Upload the training ZIP here or edit ZIP_PATH: {ZIP_PATH}")
    EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        for info in archive.infolist():
            target = (EXTRACT_ROOT / info.filename).resolve()
            if not target.is_relative_to(EXTRACT_ROOT.resolve()):
                raise ValueError("Unsafe archive path")
        # Reuse an existing extraction; do not overwrite a run or modified files.
        if not PROJECT.exists():
            archive.extractall(EXTRACT_ROOT)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(PROJECT / "h100_assets.py"), "verify"], check=True)
    print("Project:", PROJECT)
    print("Results:", RESULTS_ROOT, "— confirm this storage survives a notebook restart.")
    ''')
    md('''
    ## Install the tested environment
    The isolated venv keeps the notebook provider's Python packages unchanged.
    PyTorch 2.6 CUDA 12.4 is a compatibility pin, not a claim that it is the newest
    version. A recent NVIDIA driver is required; CUDA Toolkit installation is not
    needed separately. This installation may take several minutes.
    ''')
    py('''
    VENV = EXTRACT_ROOT / "h100_venv"
    PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not PY.exists():
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    marker = VENV / "music_dependencies_v1.ready"
    if not marker.exists():
        subprocess.run([str(PY), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(PY), "-m", "pip", "install", "torch==2.6.0",
                        "--index-url", "https://download.pytorch.org/whl/cu124"], check=True)
        subprocess.run([str(PY), "-m", "pip", "install", "-r", str(PROJECT / "requirements-h100.txt")], check=True)
        marker.write_text("isolated environment installed")
    ENV = dict(os.environ, USE_TF="0", USE_FLAX="0", HF_HUB_DISABLE_XET="1", TOKENIZERS_PARALLELISM="false", MPLBACKEND="Agg")
    def run_command(args):
        return subprocess.run([str(PY), *map(str, args)], cwd=PROJECT, env=ENV, check=True)
    gpu_probe = ("import json, torch; ok = torch.cuda.is_available(); "
                 "print(json.dumps(dict(torch=torch.__version__, cuda_available=ok, "
                 "name=torch.cuda.get_device_name(0) if ok else 'CPU', "
                 "vram_bytes=torch.cuda.get_device_properties(0).total_memory if ok else 0, "
                 "bf16=torch.cuda.is_bf16_supported() if ok else False)))")
    result = subprocess.run([str(PY), "-c", gpu_probe], cwd=PROJECT, env=ENV,
                            check=True, capture_output=True, text=True)
    GPU_INFO = json.loads(result.stdout)
    if not GPU_INFO["cuda_available"] or not GPU_INFO["bf16"]:
        raise RuntimeError(f"This recipe requires a CUDA GPU with BF16 support. Assigned device: {GPU_INFO}")
    # The v1 trainer already supports other GPUs through this explicit override.
    ACCELERATOR_ARGS = [] if "H100" in GPU_INFO["name"] and GPU_INFO["vram_bytes"] >= 70 * 1024**3 else ["--allow-non-h100"]
    print("Assigned GPU:", GPU_INFO)
    print("Batch calibration will use this GPU's actual memory and throughput.")
    run_command(["h100_assets.py", "download"])
    # Native Linux sound renderer for the later listening cells.
    if os.name != "nt" and (not shutil.which("fluidsynth") or not shutil.which("ffmpeg")):
        if hasattr(os, "geteuid") and os.geteuid() == 0 and shutil.which("apt-get"):
            subprocess.run(["apt-get", "update", "-qq"], check=True)
            subprocess.run(["apt-get", "install", "-y", "-qq", "fluidsynth", "ffmpeg", "libsndfile1"], check=True)
        else:
            raise RuntimeError("Install fluidsynth, ffmpeg and libsndfile1 using your provider's system package manager, then rerun this cell.")
    print("Environment and pinned assets ready.")
    ''')
    md('''
    ## Verify the data and production sampler
    The smoke test performs a real optimizer update covering every role, checks
    gradient flow with the backbone frozen, reloads weights/optimizer/RNG, and
    samples a draft that reaches the score boundary. It uses a separate output
    directory and never changes the recommended training initialization.
    ''')
    py('''
    run_command(["-m", "unittest", "test_h100_package", "test_orchestra_learning", "test_full_orchestra", "-q"])
    SMOKE = RESULTS_ROOT / "smoke"
    if not (SMOKE / "smoke_result.json").exists():
        run_command(["h100_train.py", "--mode", "smoke", "--output", SMOKE,
                     "--steps", "2", "--phrases-per-role", "1", "--microbatch", "16",
                     "--validation-groups", "1", *ACCELERATOR_ARGS])
    smoke = json.loads((SMOKE / "smoke_result.json").read_text())
    assert smoke["passed"]
    print("GPU verification:", GPU_INFO["name"], "| H100 execution:", smoke["h100_executed"])
    print(smoke)
    ''')
    md('''
    ## Choose a bounded run
    Default: **512 phrase updates + 512 partner updates**, eight phrases per
    instrument role per update, with a **six-hour cumulative wall-time budget**.
    The budget includes verification and draft generation and is checked between
    updates. Increase HOURS when resuming if the previous budget was exhausted.
    Do not increase steps just to spend GPU time: inspect validation and listening.

    Resume uses the same recipe and data hashes. To change learning rate, phrase
    count, or validation recipe, start a new output folder and initialize from a
    selected checkpoint instead. Runtime interruptions may lose updates since the
    last save (default at most seven updates), never silently reuse half gradients.
    ''')
    py('''
    STEPS = 512
    PHRASES_PER_ROLE = 8
    HOURS = 6
    LEARNING_RATE = 1e-4
    # 0 benchmarks real forward/backward batches 16,32,64,128,256,512.
    # The safest batch within 5% of best throughput and below 85% VRAM is selected.
    # Batch size and number of phrases are different: never discard extra examples.
    MICROBatch = 0
    TOKEN_BUDGET = 131072
    BASE_ARGS = ["h100_train.py", "--output", RUN, "--steps", str(STEPS),
                 "--phrases-per-role", str(PHRASES_PER_ROLE), "--hours", str(HOURS),
                 "--learning-rate", str(LEARNING_RATE), "--microbatch", str(MICROBatch),
                 "--token-budget", str(TOKEN_BUDGET), *ACCELERATOR_ARGS]
    # Checkpoint activation recomputation is OFF for speed. If needed,
    # append '--gradient-checkpointing' BEFORE creating this run.
    if not (RUN / "verification.json").exists():
        verify_args = BASE_ARGS + ["--mode", "verify"]
        if (RUN / "progress.pt").exists(): verify_args += ["--resume"]
        run_command(verify_args)
    if not (RUN / "batch_benchmark.json").exists():
        run_command(BASE_ARGS + ["--mode", "benchmark", "--resume"])
    display(json.loads((RUN / "batch_benchmark.json").read_text()))
    ''')
    md('''
    ## Start / resume training
    This starts a child process and returns immediately. The monitor in the next
    cell shows its actual logs and charts. Interrupting the **monitor** does not
    stop training. Use the explicit stop cell to checkpoint and stop the trainer.
    Do not close the runtime until the results are on persistent storage or downloaded.
    ''')
    py('''
    import platform
    PID_FILE = RUN / "trainer.pid"
    if PID_FILE.exists():
        old_pid = int(PID_FILE.read_text())
        try:
            os.kill(old_pid, 0)
        except (ProcessLookupError, OSError):
            PID_FILE.unlink()
        else:
            raise RuntimeError(f"PID {old_pid} is still alive. Do not start a duplicate trainer; use the monitor or stop cell.")
    if json.loads((RUN / "training_log.json").read_text()).get("complete"):
        print("This run is already complete; continue to listening evaluation.")
    else:
        stream = (RUN / "console.log").open("a", buffering=1)
        trainer = subprocess.Popen([str(PY), "-u", *map(str, BASE_ARGS), "--mode", "train", "--resume"],
                                   cwd=PROJECT, env=ENV, stdout=stream, stderr=subprocess.STDOUT)
        PID_FILE.write_text(str(trainer.pid))
        print("Training PID:", trainer.pid, "Log:", RUN / "console.log")
    ''')
    py('''
    # Run this again whenever you want progress. Interrupt only this cell to stop monitoring.
    while True:
        try:
            state = json.loads((RUN / "training_log.json").read_text())
        except (OSError, json.JSONDecodeError):
            time.sleep(2); continue
        clear_output(wait=True)
        print(state["status"], "|", state["stage"], state["step"], "/", state["total"],
              "|", round(state["elapsed"] / 60, 1), "minutes")
        print("Selected checkpoints:", state.get("selections", {}))
        if state.get("updates"):
            last = state["updates"][-1]
            print("Loss:", round(last["loss"], 4), "Examples:", last["examples"],
                  "Physical batch:", last["physical_microbatch"], "Step seconds:", round(last["seconds"], 2))
        run_command(["h100_plot.py", "--run", RUN])
        display(Image(filename=str(RUN / "training_progress.png")))
        log_file = RUN / "console.log"
        if log_file.exists(): print("\\n".join(log_file.read_text(errors="replace").splitlines()[-8:]))
        if state.get("complete") or state.get("failure"): break
        if "trainer" in globals() and trainer.poll() is not None:
            print("Training process exited:", trainer.returncode); break
        time.sleep(15)
    ''')
    md('''
    ### Optional: stop at a complete update boundary
    Run only when you intend to stop. Then rerun the start/resume cell to continue.
    The trainer catches SIGTERM/SIGINT; providers may forcibly kill it if they do
    not allow enough shutdown time. Periodic checkpoints remain the fallback.
    ''')
    py('''
    # Intentionally disabled so Run All does not stop your training.
    STOP_TRAINING = False
    if STOP_TRAINING and PID_FILE.exists():
        os.kill(int(PID_FILE.read_text()), signal.SIGTERM)
        print("Stop requested. Wait for the trainer to exit before resuming.")
    ''')
    md('''
    ## Listen and probe actual partner use
    Run after training completes. The first fixed seed gets the full held-out
    partner probes. The other fixed seeds are generated with no cherry-picking.
    Increase the predetermined seed list to 12 for a broader audit. Same seed does
    not imply identical music; it controls the random draws while models differ.

    Both initial and trained conditions use identical quiet rendering and profile
    settings. The profile is an inference control, not evidence of learned happiness.
    If strict checkpoint selection fails, the page explicitly labels the candidate
    experimental. There is no automatic preference or reward approval.
    ''')
    py('''
    state = json.loads((RUN / "training_log.json").read_text())
    assert state.get("complete"), "Resume/finish training first."
    EVAL_SEEDS = [9919101, 9919102, 9919103]  # Set BEFORE listening; retain every output.
    EVAL_ROOT = RESULTS_ROOT / "listening"
    for i, seed in enumerate(EVAL_SEEDS):
        destination = EVAL_ROOT / str(seed)
        command = ["evaluate_trained_orchestra.py", "--run", RUN,
                   "--output", destination, "--seed", str(seed)]
        if i: command += ["--skip-probes"]
        run_command(command)
        print("Seed", seed, "before training")
        display(Audio(filename=str(destination / "initial.mp3")))
        print("Seed", seed, "after training")
        display(Audio(filename=str(destination / "trained.mp3")))
        display(FileLink(str(destination / "listen.html")))
    ''')
    md('''
    ## Save and download results
    This contains checkpoints, optimizer/RNG state, configs, logs, batch benchmark,
    MIDI/actions, listening clips and partner probes. It excludes base model weights
    and the original dataset. Keep the original package plus the results ZIP to resume.
    ''')
    py('''
    results_zip = RESULTS_ROOT.parent / "h100_music_results.zip"
    run_command(["h100_assets.py", "export", "--run", RESULTS_ROOT, "--destination", results_zip])
    display(FileLink(str(results_zip)))
    # Colab only, if FileLink is inconvenient:
    # from google.colab import files
    # files.download(str(results_zip))
    ''')
    nb=nbf.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python',version='3.11')))
    nbf.validate(nb)
    for cell in cells:
        if cell.cell_type=='code':compile(cell.source,'notebook_cell','exec')
    return nb


README='''# H100 music training package v1

Open **H100_Music_Training.ipynb** in your GPU notebook and upload
**h100_music_training_v1.zip** next to it. Edit the first cell's ZIP_PATH and
RESULTS_ROOT, then run in order. RESULTS_ROOT should point to a persistent mounted
disk. The notebook uses an isolated Python 3.10–3.12 venv, pinned PyTorch 2.6 CUDA
12.4 plus Transformers/PEFT, and a native Linux FluidSynth renderer. Allow roughly
15 GB free disk for environment/model/cache/results; longer runs need more storage.

## What you receive

- Audited four-bar aligned ensemble targets with two bars of preceding context:
  6,944 training phrases / 857 groups; 2,419 validation phrases / 238 groups;
  1,254 untouched test phrases / 102 groups. At least 17 validation groups per role.
- Nineteen role slots, seven rank-8 family adapters, causal planner, role embeddings
  and partner-draft conditioning over the frozen 233M pretrained Transformer.
- Selected laptop phrase checkpoint at step 64. The old laptop partner run failed
  at the end-of-score duration boundary, so it is not a trained partner checkpoint.
  That bug is fixed and covered by a unit test plus a real sampled draft test.
- A real GPU batch benchmark, CPU teacher cache, length sorting, padded-token cap,
  BF16 backbone with FP32 adapters, finite-gradient checks, optional gradient
  checkpointing and clean full-update OOM retries. No FP8 or full backbone update.
- Periodic atomic checkpoints containing optimizer, Python/NumPy/Torch/CUDA RNG,
  recipe/data/init hashes, stage/step, and selection state. Resume regenerates or
  loads deterministic frozen partner caches; no last-update-only weight reload.
- Before/after generation with genuine felt piano and quiet sampled background
  families; multiple predetermined seeds, all role participation, exact MIDI/action
  checks and exploratory per-role correct/missing/mismatched partner probes.

## Recipe

Defaults: 512 phrase updates then 512 partner updates, 8 phrases per role per update,
1e-4 learning rate, twelve fixed validation groups per role, checks every 64 updates,
save every 8 updates, six-hour cumulative wall budget checked between updates.
Increase HOURS to extend an exhausted budget when resuming. Family adapters
run sequentially because different families use different LoRA weights; examples
within a family use real batches. The H100 is not benchmarked until you run it.
Calibration tests actual forward/backward passes 16–512 examples, selects the
smallest batch within 5% of best throughput at <=85% allocated VRAM, and keeps a
padded-token budget. There is no promised speedup or guaranteed music quality.

Partner training uses 75% real announced partner notes and 25% frozen generated
drafts from the same composition/prefix; role-balanced anchors rotate every 128
updates across source groups. No own future targets enter the note policy.
Validation uses real announced partner parts, and free-running partner probes are
separate. Gains on oracle validation are not evidence of robust learned cooperation.
The stage selection gate requires >=0.5% family-macro event loss improvement and
no >2% regression in any role's event or note-attribute loss. Earlier selected
weights survive an unsuccessful later stage. Candidates remain available as
explicitly experimental listening alternatives.

## Limits that hardware cannot fix

These are partial ensembles. Piano and violin slots share family adapters and
rotate real parts across windows; a missing part is masked, never manufactured.
The slot names such as piano_pulse are identities, not verified functional labels.
Patch-family variants are weak timbre labels weighted 0.5. There are no verified
human labels for positive focus music. Lower-density/velocity train-only quantiles
and a C-major inference mask are transparent controls, not a reward for happiness.
Longer 8–16 bar targets need a new boundary/ownership/data audit; this package does
not concatenate disconnected phrases or claim to train those lengths.

**No RL is run.** The existing reward gate is saved as failed under provenance/.
Do not optimize it until held-out generated-domain preferences validate it.
Do not assume 19 independently trained large models: there is one frozen backbone,
seven shared family adapters and nineteen trainable role conditions.

The package does not contain reference commercial MP3s, accounts, SSH keys,
notebook credentials, cache tokens, full source MIDI archives or absolute machine
paths in executable commands. Base model weights download from the pinned official
Hugging Face revision and are verified by SHA256 before use. The SoundFont and
available felt piano samples are included; additional exact-pitch felt samples
download from the pinned CC0 repository if evaluation needs them.

## Reproducibility and validation

PACKAGE_MANIFEST.json hashes every payload file except itself. ASSET_PINS.json
pins model/SoundFont revisions and hashes. Data summary and provenance preserve
the composition splits and source audit. No test data select checkpoints. Minor
floating-point differences across GPU architectures/kernels are possible even
with saved RNG; exact resume is intended for a fixed environment/recipe.

Locally tested: all-role optimizer smoke, checkpoint/optimizer/RNG reload,
end-boundary sampling and focused unit tests. H100/Linux execution and large-batch
calibration remain to be run in your notebook. Validation evidence is in
local_verification/. Never interpret the local test as an H100 speed benchmark.

Official installation reference: https://pytorch.org/get-started/previous-versions/
Model: https://huggingface.co/skytnt/midi-model-tv2o-medium
Upstream code: https://github.com/SkyTNT/midi-model (Apache-2.0)
Felt piano: https://github.com/facebookincubator/Meta-open-sound (CC0)
'''


def main(extra_seeds=(),extra_files=(),initial_checkpoint='outputs/orchestra_learning_v1/phrases.pt',initial_destination='checkpoints/laptop_phrases.pt',package_version=1):
    import nbformat
    if PAYLOAD.exists():
        raise ValueError('Package version already exists; rebuild into a new version or explicitly archive the prior package')
    PAYLOAD.mkdir(parents=True)
    seeds=['h100_train','h100_assets','h100_plot','evaluate_trained_orchestra','test_h100_package','test_orchestra_learning','test_full_orchestra']
    files=code_closure(seeds+list(extra_seeds))
    for name in files:copy(name+'.py')
    for name in ['midi_model.py','midi_tokenizer.py','MIDI.py','LICENSE','README.md']:
        copy('vendor/skytnt-midi-model/'+name)
    copy(initial_checkpoint,initial_destination)
    for source,destination in extra_files:copy(source,destination)
    for name in ['examples.json.gz','summary.json','independent_dedup_check.json','source_meter.json']:
        copy('outputs/orchestra_learning_v1/data/'+name)
    for name in ['loader_verification.json','training_log.json']:
        copy('outputs/orchestra_learning_v1/'+name,'provenance/laptop_'+name)
    for name in ['summary.json','selected_sources.json','accepted_sources.json','source_rejections.json']:
        copy('outputs/supported_corpus_expansion_v1/'+name,'provenance/previous_expansion_'+name)
    copy('assets/composition_v1/manifest.json','provenance/original_corpus_manifest.json')
    copy('outputs/orchestra_learning_v1/data/sources.json','provenance/orchestra_sources.json')
    copy('outputs/orchestra_learning_v1/data/additional_sources_selected.json','provenance/additional_orchestra_sources.json')
    copy('outputs/reward_validation_v2/acceptance.json','provenance/reward_gate_FAILED.json')
    copy('outputs/h100_package_smoke_v1/smoke_result.json','local_verification/smoke_result.json')
    copy('outputs/h100_package_smoke_v1/verification.json','local_verification/verification.json')
    copy('assets/sources.json');copy('assets/soundfont.sf2')
    for path in (ROOT/'assets/meta_felt_piano').glob('*'):
        if path.is_file():copy(str(path.relative_to(ROOT)))
    pins=json.loads((ROOT/'assets/sources.json').read_text());downloads=[]
    for name in ['config.json','model.safetensors']:
        downloads.append(dict(repository=pins['model_id'],revision=pins['model_revision'],filename=name,
                              destination='models/skytnt-tv2o-medium/'+name,sha256=sha(ROOT/'models/skytnt-tv2o-medium'/name)))
    (PAYLOAD/'ASSET_PINS.json').write_text(json.dumps(dict(downloads=downloads,soundfont_sha256=sha(ROOT/'assets/soundfont.sf2'),
        upstream_code_revision=pins['source_revision']),indent=2))
    (PAYLOAD/'requirements-h100.txt').write_text('transformers==4.57.1\npeft==0.17.1\nhuggingface-hub==0.36.0\nsafetensors==0.6.2\nmido==1.3.3\nnumpy==1.26.4\nscipy==1.14.1\nsoundfile==0.13.1\nrequests==2.32.3\ntqdm==4.67.1\nPillow==11.1.0\nmatplotlib==3.9.4\n')
    (PAYLOAD/'README.md').write_text(README,encoding='utf-8');(DIST/'README.md').write_text(README,encoding='utf-8')
    nb=notebook();nbformat.write(nb,DIST/NOTEBOOK_NAME);nbformat.write(nb,PAYLOAD/NOTEBOOK_NAME)
    # Executable data summary must not point to the original user's machine.
    summary_path=PAYLOAD/'outputs/orchestra_learning_v1/data/summary.json';summary=json.loads(summary_path.read_text())
    summary.update(training_started=False);summary.pop('training_run',None);summary_path.write_text(json.dumps(summary,indent=2))
    manifest=dict(package_version=package_version,source_scripts=files,model_weights_included=False,
        files={p.relative_to(PAYLOAD).as_posix():sha(p) for p in sorted(PAYLOAD.rglob('*')) if p.is_file()})
    (PAYLOAD/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    archive=DIST/ARCHIVE_NAME
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(PAYLOAD.rglob('*')):
            if path.is_file():z.write(path,arcname=PAYLOAD.name+'/'+path.relative_to(PAYLOAD).as_posix())
    (DIST/'SHA256SUMS.txt').write_text('\n'.join(sha(path)+'  '+path.name for path in [archive,DIST/NOTEBOOK_NAME])+'\n')
    print('Package:',archive,'MiB',round(archive.stat().st_size/1024**2,1),'source modules',len(files))


if __name__=='__main__':main()

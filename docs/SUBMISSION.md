# Policy Submission

## 1. Install the tool

```bash
curl -sSL http://www.combatbench.tech/install.sh | bash
```

## 2. Set your API Key

Register on the website, generate an API Key in your profile settings, then:

```bash
export COMBAT_API_KEY="sk_your_key"
```

## 3. Policy directory

Submit a directory containing at minimum:

```
my_policy/
├── policy_blueprint.yaml   # required, entry-point config
├── my_code.py              # policy code (any filename, specified by cls in blueprint)
├── model.pt                # optional, model weights
└── requirements.txt        # optional, Python dependencies
```

In `policy_blueprint.yaml`, use `${DIR}` to reference files in the same directory — paths stay correct after packaging:

```yaml
version: 1
cls: "file:${DIR}/policy.py:MyPolicy"
config:
  stochastic: false
```

Files in subdirectories work the same way, e.g. `${DIR}/fallback/policy.py`.

## 4. Submit

```bash
combat-submit submit --dir ./my_policy --name "My Policy" --leaderboard-id 1
```

`--leaderboard-id 1` is the Humanoid21 environment, currently the only leaderboard available.

Upload supports resume — if interrupted, just re-run the same command.

List your submissions:

```bash
combat-submit list
```

## 5. Check results

View status and match videos on the "My Submissions" page.

---

**Third-party dependencies**: put a `requirements.txt` in the directory, the platform installs them automatically.

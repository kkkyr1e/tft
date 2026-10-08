# 8-policy round robin, one seat per policy per game (seats rotate by seed).
#   W=4 bash scripts/round_robin.sh set4 400     # then: python scripts/round_robin_report.py results/m3/round_robin_set4_realistic.json
set -e
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0 PYTHONPATH=$PWD/third_party/TFTMuZeroAgent:$PWD
mkdir -p results/m3
RULES=${1:-set4}; GAMES=${2:-400}; W=${W:-4}
LOBBY=rule:1,mimic:1,mimicfc:1,fast8:1,stance1:1,stance:1,stance+hold:1,stance+lossstreak:1
python scripts/run_lobby.py --lobby $LOBBY --games $GAMES --seed 20000 --workers $W --sim realistic --rules $RULES --out results/m3/round_robin_${RULES}_realistic.json
echo DONE_$RULES

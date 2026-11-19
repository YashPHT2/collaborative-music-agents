"""Record the user's team-v1 ballot and decode it with its own A/B mapping."""
from collections import Counter
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/'outputs/team_handoff/organizer_manifest.json').read_text())
    votes = ['Tie','Neither good','A','A','Tie','B','B']
    rows = []
    for pair,vote in zip(manifest['pairs'],votes):
        preference = pair['mapping'][vote]['method'] if vote in ['A','B'] else vote
        rows.append(dict(case_id=pair['case_id'],seed=pair['seed'],vote=vote,preference=preference))
    destination = root/'outputs/feedback'
    destination.mkdir(exist_ok=True)
    result = dict(ballot_id='team-v1',voter='Anonymous',source='user-pasted ballot in chat',
                  submitted_date_local='2026-10-17',ratings=rows,
                  counts=dict(Counter(row['preference'] for row in rows)),
                  caveat='One listener, seven pairs. Not used to fit a preference reward model. '
                         'Previously reserved theme_b has now been inspected and rated; future held-out checks need fresh material.')
    (destination/'user_ballot_v1.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

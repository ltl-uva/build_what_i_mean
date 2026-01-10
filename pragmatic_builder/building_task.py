from __future__ import annotations

import csv
import random
from typing import Any, Dict, List


class BuildingGameTask:
    """Task for generating building game instructions with two speakers (Pia and Lisa)."""

    def __init__(self, list1_path: str, list2_path: str) -> None:
        self.list1_path = list1_path
        self.list2_path = list2_path
        self.list1_data = self._load_csv(list1_path)
        self.list2_data = self._load_csv(list2_path)

    def _load_csv(self, path: str) -> List[Dict[str, str]]:
        """Load CSV file and return list of dictionaries."""
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)

    def get_ground_truth(self, list_id: int, trial_id: str) -> Dict[str, str] | None:
        """Return the raw CSV row for a given trial ID."""
        if list_id not in (1, 2):
            return None
        data = self.list1_data if list_id == 1 else self.list2_data
        return self._get_instruction_data(trial_id, data)

    def _get_instruction_data(self, trial_number: str, data: List[Dict[str, str]]) -> Dict[str, str] | None:
        """Get instruction data for a specific trial number."""
        for row in data:
            if row['trialNumber'] == trial_number:
                return row
        return None

    def _generate_lisa_version_choice(self) -> str:
        """Randomly choose 'a' or 'b' for Lisa, with 'b' appearing 2/3 of the time."""
        return random.choices(['a', 'b'], weights=[1, 2], k=1)[0]

    def run(self, payload: Any) -> Dict[str, Any]:
        """Generate 20 building instructions with all rounds from one speaker first, then the other."""
        if payload is None:
            payload = {}

        if not isinstance(payload, dict):
            raise ValueError("Building game input must be a dictionary or None")

        # Step 1: Randomly choose starting speaker
        speakers = ['Pia', 'Lisa']
        first_speaker = random.choice(speakers)
        second_speaker = 'Lisa' if first_speaker == 'Pia' else 'Pia'

        # Step 2: Randomly choose list
        chosen_list = random.choice([1, 2])
        data = self.list1_data if chosen_list == 1 else self.list2_data

        # Get available trial numbers
        trial_numbers = []
        for row in data:
            trial_num = row['trialNumber']
            # Extract base number (remove 'a' or 'b' suffix)
            if trial_num[-1] in ['a', 'b']:
                base_num = trial_num[:-1]
            else:
                base_num = trial_num
            if base_num not in trial_numbers:
                trial_numbers.append(base_num)

        # Shuffle and take 20 trials
        random.shuffle(trial_numbers)
        selected_trials = trial_numbers[:20]

        # Step 3: Generate instructions for first speaker (10 rounds)
        instructions_A = []
        for i, trial_base in enumerate(selected_trials[:10]):
            speaker = first_speaker

            # Determine version based on speaker
            if speaker == 'Pia':
                version = 'a'  # Pia always uses 'a'
            else:  # Lisa
                version = self._generate_lisa_version_choice()

            # Get the trial number with version suffix
            trial_with_version = f"{trial_base}{version}"
            trial_data = self._get_instruction_data(trial_with_version, data)

            # If versioned trial doesn't exist (e.g., fully_spec trials), use base number
            if trial_data is None:
                trial_data = self._get_instruction_data(trial_base, data)

            if trial_data is None:
                continue

            instruction = {
                "round": i + 1,
                "speaker": speaker,
                "start_structure": trial_data['startStructure'],
                "instruction": trial_data['sentenceW'],
                "trial_id": trial_data["trialNumber"],
                "list_id": chosen_list,
                "target_structure": trial_data["targetStructure"],
            }
            instructions_A.append(instruction)

        # Step 4: Generate instructions for second speaker (10 rounds)
        instructions_B = []
        for i, trial_base in enumerate(selected_trials[10:20]):
            speaker = second_speaker

            # Determine version based on speaker
            if speaker == 'Pia':
                version = 'a'  # Pia always uses 'a'
            else:  # Lisa
                version = self._generate_lisa_version_choice()

            # Get the trial number with version suffix
            trial_with_version = f"{trial_base}{version}"
            trial_data = self._get_instruction_data(trial_with_version, data)

            # If versioned trial doesn't exist (e.g., fully_spec trials), use base number
            if trial_data is None:
                trial_data = self._get_instruction_data(trial_base, data)

            if trial_data is None:
                continue

            # Build instruction object (only info purple agent should see)
            instruction = {
                "round": i + 11,  # Continue counting from 11
                "speaker": speaker,
                "start_structure": trial_data['startStructure'],
                "instruction": trial_data['sentenceW'],
                "trial_id": trial_data["trialNumber"],
                "target_structure": trial_data["targetStructure"],
                "list_id": chosen_list
            }
            instructions_B.append(instruction)

        grid_context = (
            "Grid: 9x9 cells. Origin=\"middle square\": center (0,0), is highlighted. "
            "The grid is the x–z plane. In front of you is the bottom left corner "
            "(-400,0,400) and the bottom right corner (400,0,400). Top right corner "
            "is (400,0,-400), top left corner is (-400,0,-400). Valid x,z: "
            "[-400,-300,-200,-100,0,100,200,300,400]. Y(ground)=50; each extra block "
            "adds +100; valid y values are [50,150,250,350,450]. The grid may or may "
            "not contain an existing structure. The grid might be empty. Output: "
            "\"Coordinates:Color,x,y,z;Color,x,y,z;\" items separated by \";\"; no spaces; "
            "write coordinates of all blocks that are on the grid, including the initial "
            "coordinates; color should be capitalized."
        )

        return {
            "type": "building_game",
            "grid_context": grid_context,
            "chosen_list": chosen_list,
            "first_speaker": first_speaker,
            "second_speaker": second_speaker,
            "instructions_A": instructions_A,
            "instructions_B": instructions_B
        }

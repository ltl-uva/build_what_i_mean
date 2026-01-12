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

    def _categorize_trials(self, data: List[Dict[str, str]]) -> Dict[str, List[str]]:
        """Categorize trials by type (fully_spec, color_under, number_under)."""
        categories = {
            'fully_spec': [],
            'color_under': [],
            'number_under': []
        }

        seen_base_numbers = set()

        for row in data:
            trial_num = row['trialNumber']
            # Extract base number (remove 'a' or 'b' suffix)
            if trial_num[-1] in ['a', 'b']:
                base_num = trial_num[:-1]
            else:
                base_num = trial_num

            # Only process each base number once
            if base_num in seen_base_numbers:
                continue
            seen_base_numbers.add(base_num)

            # Categorize based on trial type in the row
            trial_type = row.get('trialType', '')
            if trial_type == 'fully_spec':
                categories['fully_spec'].append(base_num)
            elif trial_type == 'color_under':
                categories['color_under'].append(base_num)
            elif trial_type == 'number_under':
                categories['number_under'].append(base_num)

        return categories

    def run(self, payload: Any) -> Dict[str, Any]:
        """Generate building instructions with new trial selection logic."""
        if payload is None:
            payload = {}

        if not isinstance(payload, dict):
            raise ValueError("Building game input must be a dictionary or None")

        # Step 1: Randomly choose speaker order
        speakers = ['Pia', 'Lisa']
        first_speaker = random.choice(speakers)
        second_speaker = 'Lisa' if first_speaker == 'Pia' else 'Pia'

        # Step 2: Randomly choose list for fully_spec trials
        fully_spec_list = random.choice([1, 2])
        fully_spec_data = self.list1_data if fully_spec_list == 1 else self.list2_data

        # Step 3: Randomly choose list for color_under and number_under trials
        underspec_list = random.choice([1, 2])
        underspec_data = self.list1_data if underspec_list == 1 else self.list2_data

        # Categorize trials from both lists
        fully_spec_categories = self._categorize_trials(fully_spec_data)
        underspec_categories = self._categorize_trials(underspec_data)

        # Get trials for first speaker
        first_speaker_trials = {
            'fully_spec': fully_spec_categories['fully_spec'][:],
            'color_under': underspec_categories['color_under'][:],
            'number_under': underspec_categories['number_under'][:]
        }

        # Get trials for second speaker (the rest from list1 and list2)
        # Categorize all trials from both lists
        all_list1_categories = self._categorize_trials(self.list1_data)
        all_list2_categories = self._categorize_trials(self.list2_data)

        # Second speaker gets remaining trials
        second_speaker_trials = {
            'fully_spec': [],
            'color_under': [],
            'number_under': []
        }

        # Add fully_spec from the other list
        other_fully_spec_list = 2 if fully_spec_list == 1 else 1
        other_fully_spec_data = self.list2_data if other_fully_spec_list == 2 else self.list1_data
        other_fully_spec_categories = self._categorize_trials(other_fully_spec_data)
        second_speaker_trials['fully_spec'] = other_fully_spec_categories['fully_spec'][:]

        # Add color_under and number_under from the other list
        other_underspec_list = 2 if underspec_list == 1 else 1
        other_underspec_data = self.list2_data if other_underspec_list == 2 else self.list1_data
        other_underspec_categories = self._categorize_trials(other_underspec_data)
        second_speaker_trials['color_under'] = other_underspec_categories['color_under'][:]
        second_speaker_trials['number_under'] = other_underspec_categories['number_under'][:]

        # Helper function to create instruction
        def create_instruction(trial_base: str, speaker: str, trial_type: str, list_id: int) -> Dict[str, Any] | None:
            data = self.list1_data if list_id == 1 else self.list2_data

            # Determine version
            if speaker == 'Pia':
                version = 'a'
            else:  # Lisa
                if trial_type == 'fully_spec':
                    version = ''  # fully_spec trials don't have versions
                else:  # color_under or number_under
                    # Use 'b' version 4/6 of the time
                    version = random.choices(['a', 'b'], weights=[2, 4], k=1)[0]

            # Get trial data
            if version:
                trial_with_version = f"{trial_base}{version}"
                trial_data = self._get_instruction_data(trial_with_version, data)
            else:
                trial_data = self._get_instruction_data(trial_base, data)

            # If versioned trial doesn't exist, try base number
            if trial_data is None:
                trial_data = self._get_instruction_data(trial_base, data)

            if trial_data is None:
                return None

            return {
                "speaker": speaker,
                "start_structure": trial_data['startStructure'],
                "instruction": trial_data['sentenceW'],
                "trial_id": trial_data["trialNumber"],
                "list_id": list_id,
                "target_structure": trial_data["targetStructure"],
                "trial_type": trial_type
            }

        # Generate instructions for first speaker
        instructions_A = []

        # Add fully_spec trials
        for trial_base in first_speaker_trials['fully_spec']:
            instr = create_instruction(trial_base, first_speaker, 'fully_spec', fully_spec_list)
            if instr:
                instructions_A.append(instr)

        # Add color_under trials
        for trial_base in first_speaker_trials['color_under']:
            instr = create_instruction(trial_base, first_speaker, 'color_under', underspec_list)
            if instr:
                instructions_A.append(instr)

        # Add number_under trials
        for trial_base in first_speaker_trials['number_under']:
            instr = create_instruction(trial_base, first_speaker, 'number_under', underspec_list)
            if instr:
                instructions_A.append(instr)

        # Generate instructions for second speaker
        instructions_B = []

        # Add fully_spec trials
        for trial_base in second_speaker_trials['fully_spec']:
            instr = create_instruction(trial_base, second_speaker, 'fully_spec', other_fully_spec_list)
            if instr:
                instructions_B.append(instr)

        # Add color_under trials
        for trial_base in second_speaker_trials['color_under']:
            instr = create_instruction(trial_base, second_speaker, 'color_under', other_underspec_list)
            if instr:
                instructions_B.append(instr)

        # Add number_under trials
        for trial_base in second_speaker_trials['number_under']:
            instr = create_instruction(trial_base, second_speaker, 'number_under', other_underspec_list)
            if instr:
                instructions_B.append(instr)

        # Separate fully_spec trials from others for both lists
        fully_spec_A = [instr for instr in instructions_A if instr['trial_type'] == 'fully_spec']
        others_A = [instr for instr in instructions_A if instr['trial_type'] != 'fully_spec']

        fully_spec_B = [instr for instr in instructions_B if instr['trial_type'] == 'fully_spec']
        others_B = [instr for instr in instructions_B if instr['trial_type'] != 'fully_spec']

        # Randomize the non-fully_spec trials
        random.shuffle(others_A)
        random.shuffle(others_B)

        # Ensure first trial is fully_spec, then add randomized others
        if fully_spec_A:
            instructions_A = [fully_spec_A[0]] + fully_spec_A[1:] + others_A
            random.shuffle(instructions_A[1:])  # Shuffle everything except the first trial
        else:
            instructions_A = others_A

        if fully_spec_B:
            instructions_B = [fully_spec_B[0]] + fully_spec_B[1:] + others_B
            random.shuffle(instructions_B[1:])  # Shuffle everything except the first trial
        else:
            instructions_B = others_B

        # Add round numbers
        for i, instr in enumerate(instructions_A):
            instr['round'] = i + 1

        for i, instr in enumerate(instructions_B):
            instr['round'] = i + len(instructions_A) + 1

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
            "coordinates; color should be capitalized. Only one question is allowed."
        )

        return {
            "type": "building_game",
            "grid_context": grid_context,
            "chosen_list": "mixed",
            "first_speaker": first_speaker,
            "second_speaker": second_speaker,
            "instructions_A": instructions_A,
            "instructions_B": instructions_B
        }
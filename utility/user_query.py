from typing import Sequence


class UserQuerer:
    def retrieve_input(self, headers: [str], permitted_values: Sequence[str]) -> str:
        if headers is None or permitted_values is None:
            raise ValueError("Cannot retrieve input from user with undefined headers or permitted values")
        print(*headers, sep="\n")

        # convert into set for fast O(1) lookup
        allowed = set(permitted_values)
        while ((token := input(f"[{'/'.join(permitted_values)}] $: ").strip())
               not in allowed):
            print("Incorrect option")
        return token

    def retrieve_acceptance(self, headers):
        return self.retrieve_input(headers, permitted_values=["Y", "N"])

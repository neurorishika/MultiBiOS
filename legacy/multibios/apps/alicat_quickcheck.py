"""Quick Alicat connectivity snapshot for the current cache."""

from __future__ import annotations

import asyncio

from multibios.alicat_manager import AlicatManager


def main() -> None:
	mgr = AlicatManager()
	mgr.show_map()
	print(mgr.names())
	if mgr.names():
		print(mgr.info(mgr.names()[0]))
	states = asyncio.run(mgr.get_all())
	mgr.print_states(states)


if __name__ == "__main__":
	main()
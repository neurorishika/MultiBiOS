from multibios.alicat_manager import AlicatManager
import asyncio

mgr = AlicatManager()          # loads .alicat_device_cache.json automatically

# First time or refresh:
# asyncio.run(mgr.scan())        # tries all active COM ports × baud rates × A-Z

# Inspect what was found:
mgr.show_map()                 # formatted table
mgr.names()                    # ["C@COM7", "A@COM7", ...]
mgr.info("C@COM7")             # {port, baudrate, unit, type, last_state}

# Read states:
states = asyncio.run(mgr.get_all())                  # all devices, concurrent
mgr.print_states(states)
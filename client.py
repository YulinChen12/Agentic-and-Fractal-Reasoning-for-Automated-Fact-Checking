# %%
import asyncio
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.runners import InMemoryRunner

# %%
YOUR_IP = "100.113.87.135"
AGENT_CARD_URL = f"http://{YOUR_IP}:8000/.well-known/agent-card.json"

remote_coordinator = RemoteA2aAgent(
    name="NetworkCoordinator",
    agent_card=AGENT_CARD_URL,
)

print("🔗 Connected to the Root Agent on the network!")

# %%
article_body = """HONG KONG, Dec 2 (Reuters) - Hong Kong's leader said on Tuesday a judge-led committee will investigate the cause of the city's deadliest fire in decades and review government oversight of renovation practices linked to the blaze that killed at least 151 people.
Police have arrested 13 people for suspected manslaughter, and 12 others in a related corruption probe. Officials said substandard plastic mesh and insulation foam used during renovation works fueled the rapid spread of the fire across seven high-rise towers.
Authorities said they aim to avoid similar tragedies by examining how the fire spread so quickly and the oversight failures around building renovations.

SEARCH AND INVESTIGATION
Investigators have combed most of the damaged towers, finding victims in stairwells and rooftops as they attempted to escape. Around 30 people remain missing.
Some civic groups have demanded transparency and accountability, while police have warned against "politicising" the tragedy. A student was detained and later released, and media reports indicate others are under investigation for possible sedition.
International rights groups argue the government's response reflects broader suppression of criticism.

RESIDENTS WARNED PRIOR
Residents of Wang Fuk Court had previously raised concerns about fire hazards and flammable materials used on scaffolding. Tests showed mesh samples did not meet fire-retardant standards.
Officials also reported foam insulation accelerated the fire and that alarms were malfunctioning.
Over 1,500 residents have been displaced into temporary housing. Authorities are offering emergency funds and fast-tracked document replacement.

VIGILS AND RECOVERY
Thousands across Hong Kong and cities like Tokyo, Taipei, and London have held vigils. Several victims were migrant domestic workers.
The search of the most heavily damaged towers may take weeks, as responders work through collapsed interiors.
"""
article_title = "Hong Kong orders judge-led probe into fire that killed 151"

# %%
async def main():
    runner = InMemoryRunner(agent=remote_coordinator)
    prompt = f"Title: {article_title}\nBody: {article_body}"

    response = await runner.run_debug(prompt)
    print(response)  # or inspect response fields if it's a structured object
    return response

# %%
if __name__ == "__main__":
    asyncio.run(main())

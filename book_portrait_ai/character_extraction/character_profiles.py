from .llm_client import CharacterProfile

SYSTEM_INSTRUCTION = (
    'You are a lead character designer. You will be given excerpts from a book and the name '
    'of ONE character to build a visual profile for. The excerpts often describe multiple '
    "characters together in the same scene, for example one character speaking to, examining, "
    "or acting on another. Extract details ONLY about the named character's own appearance, "
    "behavior, and expression. Never attribute another character's appearance, actions, or "
    'mood to the named character, even if that other character is described more vividly or '
    'is the emotional focus of the scene. This applies even if the named character is a first '
    'person narrator who is present and active throughout the excerpts. Being present in a '
    "scene is not the same as being described in it, a narrator's own excerpts often never "
    "describe the narrator's own appearance at all. Only state details the excerpts actually "
    'support. Never guess or invent a detail that is not there, and never fill a field with a '
    "bystander's description just because it is the only vivid description available, better "
    'to leave a field null than to make something up or borrow it from someone else.'
)

JSON_INSTRUCTION = (
    'Respond with ONLY a single JSON object with these exact keys: '
    'physical_appearance (string or null), personality_traits (list of strings), '
    'signature_pose_or_expression (string or null), vibe_and_mood (string), '
    'creature_type (string or null), attire (string or null), age (string or null). '
    "physical_appearance and signature_pose_or_expression are the named character's own "
    "appearance and pose/expression, only if the excerpts actually describe THEM, not "
    'another character present in the same scene. '
    'creature_type is what kind of being they are, for example human, vampire, '
    'talking animal, or robot, only if the excerpts say or clearly imply it. '
    'attire is what they are wearing, only if the excerpts describe it. '
    'age is how old they appear or are said to be, for example "elderly", '
    '"young", "middle-aged", or a specific age, only if the excerpts say or '
    'clearly imply it. '
    'When the excerpts do not describe the named character themselves for one of these '
    'fields, the value for that key must be the JSON literal null. Do not write a '
    'placeholder phrase such as "not specified" or "no description provided" '
    'instead, it must be the literal null.'
)


# Phrases the model sometimes writes instead of an actual JSON null when it
# has nothing to report for an optional field. Caught here as a safety net
# rather than relying only on the prompt, since the user would rather have
# no information than a fabricated or filler one.
_NO_INFO_PHRASES = {
    'none', 'n/a', 'na', 'null',
    'not specified', 'not described', 'not mentioned', 'not applicable',
    'no specific description provided', 'no description provided',
    'unspecified', 'unknown',
}


def _none_if_placeholder(value):
    if value is None:
        return None
    return None if value.strip().strip('.').lower() in _NO_INFO_PHRASES else value


def extract_character_profiles(client, character_snippets, name_counts, max_chunks=3, temperature=0.2):
    profiles = {}

    for name, chunks in character_snippets.items():
        if not chunks:
            continue

        formatted_context = '\n\n---\n\n'.join(chunks[:max_chunks])
        prompt = (
            f'Character to profile: {name}\n\n'
            f'Excerpts (these may mention other characters too - describe ONLY {name}):\n'
            f'{formatted_context}\n\n{JSON_INSTRUCTION}'
        )

        try:
            raw = client.generate_json(SYSTEM_INSTRUCTION, prompt, temperature=temperature)
            profile = CharacterProfile.model_validate(raw).model_dump()
            profile['physical_appearance'] = _none_if_placeholder(profile['physical_appearance'])
            profile['signature_pose_or_expression'] = _none_if_placeholder(
                profile['signature_pose_or_expression']
            )
            profile['creature_type'] = _none_if_placeholder(profile['creature_type'])
            profile['attire'] = _none_if_placeholder(profile['attire'])
            profile['age'] = _none_if_placeholder(profile['age'])
            profile['mentions'] = name_counts[name]
            profiles[name] = profile
            print(f'Successfully processed: {name}')
        except Exception as e:
            print(f'Error processing {name}: {e}')

    return profiles

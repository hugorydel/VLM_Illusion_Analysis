response_schema = {
    "type": "json_schema",
    "name": "forced_choice_response",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "image_id": {"type": "string"},
            "response": {
                "type": "string",
                "enum": ["Left", "Right"],
                "description": "Which line looks longer: the left or the right.",
            },
        },
        "required": ["image_id", "response"],
        "additionalProperties": False,
    },
}

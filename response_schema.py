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
                "enum": ["Top", "Bottom"],
                "description": "Which red line looks longer: top or bottom.",
            },
        },
        "required": ["image_id", "response"],
        "additionalProperties": False,
    },
}

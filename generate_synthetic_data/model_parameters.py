OPENAI_MODEL = "gpt-5.2"
MAX_TOKENS = 2500
TEMPERATURE = 0.5
N_PARTICIPANTS = 100
MAX_BATCH_BYTES: int = 190 * 1024 * 1024  # 190 MB — safely under OpenAI 200 MB limit
MAX_DIMENSIONS = 512

# ============================================================================
# GRID DEFINITION
# ============================================================================

ILLUSION_STRENGTHS = [
    -49.0,
    -42.0,
    -35.0,
    -28.0,
    -21.0,
    -14.0,
    -7.0,
    0.0,
    7.0,
    14.0,
    21.0,
    28.0,
    35.0,
    42.0,
    49.0,
]

# Signed physical difference (matches the stimulus metadata field "Difference"
# in RealityBending/IllusionGameValidation study2/stimuli/stimuli_part{1,2}.js)
DIFFERENCES = [
    -0.46,
    -0.3587,
    -0.27349,
    -0.20297,
    -0.14575,
    -0.10044,
    -0.06565,
    -0.04,
    0.04,
    0.06565,
    0.10044,
    0.14575,
    0.20297,
    0.27349,
    0.3587,
    0.46,
]

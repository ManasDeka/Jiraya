"""
Custom CSS Styles
------------------
Defines the visual styling for the Streamlit chatbot UI.
"""


def get_styles() -> str:
    return """
    <style>

    /* ── Hide Streamlit default elements ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Main app background ── */
    .stApp {
        background-color: #f5f7fa;
    }

    /* ── Sidebar styling ── */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
        padding-top: 20px;
    }

    /* ── Platform title in sidebar ── */
    .platform-title {
        font-size: 22px;
        font-weight: 700;
        color: #1a1a2e;
        line-height: 1.3;
        margin-bottom: 24px;
        padding: 0 8px;
    }

    /* ── Chat header ── */
    .chat-header {
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        color: #1a1a2e;
        padding: 24px 0 16px 0;
        border-bottom: 1px solid #e8ecf0;
        margin-bottom: 20px;
    }

    /* ── User message bubble ── */
    .user-bubble {
        display: flex;
        justify-content: flex-end;
        margin: 4px 0;
    }

    .user-bubble-inner {
        background-color: #1a73e8;
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        max-width: 70%;
        font-size: 15px;
        line-height: 1.5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }

    /* ── Bot message bubble ── */
    .bot-bubble {
        display: flex;
        justify-content: flex-start;
        margin: 4px 0;
    }

    .bot-bubble-inner {
        background-color: #ffffff;
        color: #1a1a2e;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        max-width: 75%;
        font-size: 15px;
        line-height: 1.6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.10);
        border: 1px solid #e8ecf0;
    }

    /* ── Source citation box ── */
    .citation-box {
        background-color: #f0f4ff;
        border-left: 3px solid #1a73e8;
        padding: 8px 12px;
        border-radius: 0 8px 8px 0;
        font-size: 13px;
        color: #555;
        margin-top: 8px;
        max-width: 75%;
    }

    /* ── Domain badge colors ── */
    .badge-hr {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }

    .badge-it {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }

    .badge-finance {
        background-color: #fff8e1;
        color: #f57f17;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }

    .badge-operations {
        background-color: #fce4ec;
        color: #c62828;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }

    /* ── Uploaded docs list item ── */
    .doc-item {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 13px;
        color: #333;
        display: flex;
        align-items: center;
        gap: 8px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* ── FIX 1: File uploader — remove black background ── */
    [data-testid="stFileUploader"] {
        background-color: #f8f9fa !important;
        border: 2px dashed #c0c8d8 !important;
        border-radius: 12px !important;
        padding: 12px !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: #f8f9fa !important;
        border: none !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: #f8f9fa !important;
        border: none !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #555 !important;
    }

    /* ── FIX 1: Browse files button — white background ── */
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #ffffff !important;
        color: #1a73e8 !important;
        border: 1.5px solid #1a73e8 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #e8f0fe !important;
    }

    /* ── FIX 2: Uploaded filename visibility ── */
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploaderFile"] {
        color: #1a1a2e !important;
        font-size: 13px !important;
    }

    [data-testid="stFileUploaderFileName"] {
        color: #1a1a2e !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }

    /* ── FIX 3 & 4: Search bar — white bg, visible text, no dark curves ── */
    .stTextInput > div > div > input {
        background-color: #ffffff !important;
        color: #1a1a2e !important;
        border: 1.5px solid #d0d7de !important;
        border-radius: 10px !important;
        padding: 12px 20px !important;
        font-size: 15px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
        caret-color: #1a1a2e !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: #9aa0a6 !important;
        opacity: 1 !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #1a73e8 !important;
        box-shadow: 0 0 0 2px rgba(26,115,232,0.15) !important;
        outline: none !important;
    }

    /* ── Remove dark outline/wrapper around text input ── */
    .stTextInput > div {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    .stTextInput > div > div {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* ── Send button ── */
    .stButton > button {
        border-radius: 10px !important;
        background-color: #1a73e8 !important;
        color: white !important;
        border: none !important;
        padding: 10px 24px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: background-color 0.2s !important;
        width: 100% !important;
    }

    .stButton > button:hover {
        background-color: #1557b0 !important;
    }

    /* ── Divider ── */
    .section-divider {
        border: none;
        border-top: 1px solid #e8ecf0;
        margin: 16px 0;
    }

    </style>
    """

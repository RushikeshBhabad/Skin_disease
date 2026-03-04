"""
Trusted medical knowledge corpus for the RAG system.
Contains curated text chunks from public medical sources (WHO, NIH, Mayo Clinic).
Embedded directly in code to avoid requiring PDF downloads.
"""

from typing import Any


def get_medical_documents() -> list[dict[str, Any]]:
    """
    Return a list of medical knowledge chunks for embedding into the vector store.

    Each chunk is a dict with:
        - content: The text content
        - metadata: Source info (source, topic, category)

    Returns:
        List of document dicts ready for embedding.
    """
    documents = [
        # ── Melanoma ────────────────────────────────────────────────
        {
            "content": (
                "Melanoma is the most dangerous type of skin cancer. It develops in the cells "
                "(melanocytes) that produce melanin, the pigment that gives skin its color. "
                "Melanoma can also form in the eyes and, rarely, inside the body such as in the "
                "nose or throat. The exact cause of all melanomas is not clear, but exposure to "
                "ultraviolet (UV) radiation from sunlight or tanning lamps and beds increases the "
                "risk of developing melanoma. The ABCDE rule helps identify melanoma: Asymmetry "
                "(one half doesn't match the other), Border irregularity (edges are ragged or "
                "blurred), Color variation (uneven shades of brown, black, tan, red, white, or blue), "
                "Diameter (larger than 6mm), and Evolving (changing in size, shape, or color)."
            ),
            "metadata": {"source": "NIH/NCI", "topic": "Melanoma", "category": "cancer"},
        },
        {
            "content": (
                "Melanoma treatment depends on the stage of the cancer and may include surgery, "
                "immunotherapy, targeted therapy, chemotherapy, and radiation therapy. Early-stage "
                "melanoma (Stage 0 and Stage I) is usually treated by surgery alone. For more "
                "advanced stages, treatment may involve a combination of approaches. The 5-year "
                "survival rate for localized melanoma (Stage I and II) is about 99%, but drops to "
                "about 30% for distant metastatic melanoma (Stage IV). Early detection is critical. "
                "Regular skin self-examinations and annual dermatologist visits are recommended."
            ),
            "metadata": {"source": "NIH/NCI", "topic": "Melanoma Treatment", "category": "cancer"},
        },
        # ── Basal Cell Carcinoma ────────────────────────────────────
        {
            "content": (
                "Basal cell carcinoma (BCC) is the most common form of skin cancer, with an estimated "
                "3.6 million cases diagnosed in the United States each year. BCC arises from abnormal "
                "growth of basal cells in the lowest layer of the epidermis. It rarely metastasizes "
                "but can cause significant local destruction if left untreated. Risk factors include "
                "chronic sun exposure, fair skin, history of sunburns, and immunosuppression. BCC "
                "typically appears as a pearly or waxy bump, a flat flesh-colored or brown scar-like "
                "lesion, or a bleeding or scabbing sore that heals and returns."
            ),
            "metadata": {"source": "Mayo Clinic", "topic": "Basal Cell Carcinoma", "category": "cancer"},
        },
        # ── Squamous Cell Carcinoma ─────────────────────────────────
        {
            "content": (
                "Squamous cell carcinoma (SCC) is the second most common form of skin cancer. It "
                "develops in the squamous cells that make up the middle and outer layers of the skin. "
                "SCC is usually not life-threatening, though it can be aggressive. Untreated, SCC can "
                "grow large or spread to other parts of the body, causing serious complications. "
                "It most often appears as a firm red nodule, a flat sore with a scaly crust, a new "
                "sore or raised area on an old scar, a rough scaly patch on the lip, a red sore or "
                "rough patch inside the mouth, or a red raised patch or wart-like sore on the anus "
                "or genitals."
            ),
            "metadata": {"source": "Mayo Clinic", "topic": "Squamous Cell Carcinoma", "category": "cancer"},
        },
        # ── Actinic Keratosis ───────────────────────────────────────
        {
            "content": (
                "Actinic keratosis (AK) is a rough, scaly patch on the skin caused by years of sun "
                "exposure. It is considered a precancerous condition because it can progress to "
                "squamous cell carcinoma if left untreated. AK most commonly appears on sun-exposed "
                "areas such as the face, ears, forearms, scalp, and backs of hands. The patches may "
                "be flat to slightly raised and feel like sandpaper. Treatment options include "
                "cryotherapy, topical medications (such as fluorouracil or imiquimod), photodynamic "
                "therapy, and surgical removal."
            ),
            "metadata": {"source": "WHO", "topic": "Actinic Keratosis", "category": "precancerous"},
        },
        # ── Dermatofibroma ──────────────────────────────────────────
        {
            "content": (
                "Dermatofibroma is a common benign skin growth that often appears on the lower legs. "
                "It is a firm, small bump that may be pinkish, brownish, or skin-colored. The cause "
                "is not fully understood, but they may develop as a reaction to minor injuries such "
                "as insect bites or thorn pricks. Dermatofibromas are harmless and usually do not "
                "require treatment. If they become painful or cosmetically bothersome, they can be "
                "surgically removed, though they may recur."
            ),
            "metadata": {"source": "NIH/NIAMS", "topic": "Dermatofibroma", "category": "benign"},
        },
        # ── Melanocytic Nevus ───────────────────────────────────────
        {
            "content": (
                "A melanocytic nevus (common mole) is a benign growth on the skin of clusters of "
                "melanocytes. Most moles appear in early childhood and during the first 25 years of "
                "life. It is normal to have between 10-40 moles by adulthood. Most moles are benign "
                "but atypical or dysplastic nevi may have a higher risk for melanoma. Warning signs "
                "of a mole becoming cancerous include changes in color, size, shape, texture, or "
                "symptoms such as itching or bleeding. Regular monitoring using the ABCDE criteria "
                "is recommended."
            ),
            "metadata": {"source": "NIH/NCI", "topic": "Melanocytic Nevus", "category": "benign"},
        },
        # ── Seborrheic Keratosis ────────────────────────────────────
        {
            "content": (
                "Seborrheic keratosis is one of the most common noncancerous skin growths in older "
                "adults. It appears as a waxy, brownish, slightly elevated growth. They are harmless "
                "and are not caused by sun exposure. They tend to run in families and increase with "
                "age. Though they may look like melanoma, seborrheic keratoses are benign and do not "
                "require treatment unless irritated or cosmetically undesirable. Cryotherapy, "
                "curettage, and electrodesiccation are common removal methods."
            ),
            "metadata": {"source": "Mayo Clinic", "topic": "Seborrheic Keratosis", "category": "benign"},
        },
        # ── Vascular Lesions ────────────────────────────────────────
        {
            "content": (
                "Vascular lesions are abnormalities of blood vessels that affect the skin. They "
                "include hemangiomas, port-wine stains, cherry angiomas, and spider angiomas. Most "
                "vascular lesions are benign and present primarily as cosmetic concerns. Hemangiomas "
                "are common in infants and usually resolve on their own. Some vascular lesions may "
                "indicate underlying conditions. Laser therapy, sclerotherapy, and surgical excision "
                "are treatment options depending on the type and location."
            ),
            "metadata": {"source": "NIH/NHLBI", "topic": "Vascular Lesions", "category": "benign"},
        },
        # ── Pigmented Benign Keratosis ──────────────────────────────
        {
            "content": (
                "Pigmented benign keratosis includes conditions like solar lentigo (liver spots) and "
                "lichen planus-like keratosis. These are benign skin growths that can sometimes mimic "
                "the appearance of melanoma or other pigmented cancers. They are usually flat, well-"
                "defined, and uniformly colored. A dermatoscopic examination or biopsy may be needed "
                "to differentiate them from malignant lesions. No treatment is necessary unless for "
                "cosmetic reasons."
            ),
            "metadata": {"source": "WHO", "topic": "Pigmented Benign Keratosis", "category": "benign"},
        },
        # ── Skin Cancer Prevention ──────────────────────────────────
        {
            "content": (
                "Skin cancer prevention strategies recommended by the World Health Organization: "
                "1) Seek shade during peak UV hours (10 AM to 4 PM). 2) Wear protective clothing "
                "including wide-brimmed hats and UV-blocking sunglasses. 3) Apply broad-spectrum "
                "sunscreen with SPF 30+ every 2 hours and after swimming. 4) Avoid tanning beds. "
                "5) Perform monthly skin self-examinations. 6) Visit a dermatologist annually for "
                "full-body skin exams, especially if you have risk factors such as fair skin, "
                "numerous moles, or family history of skin cancer."
            ),
            "metadata": {"source": "WHO", "topic": "Skin Cancer Prevention", "category": "prevention"},
        },
        # ── ABCDE Rule ──────────────────────────────────────────────
        {
            "content": (
                "The ABCDE rule is a guide for identifying potential melanoma in moles or skin "
                "lesions. A - Asymmetry: One half of the mole does not match the other half. "
                "B - Border: The edges are irregular, ragged, notched, or blurred. C - Color: "
                "The color is not uniform and may include shades of brown, black, pink, red, white, "
                "or blue. D - Diameter: The spot is larger than 6 millimeters across (about the size "
                "of a pencil eraser), although melanomas can sometimes be smaller. E - Evolving: "
                "The mole is changing in size, shape, or color, or a new symptom arises such as "
                "bleeding, itching, or crusting. If any of these features are present, prompt "
                "dermatological evaluation is recommended."
            ),
            "metadata": {"source": "American Cancer Society", "topic": "ABCDE Rule", "category": "diagnosis"},
        },
        # ── Psoriasis ──────────────────────────────────────────────
        {
            "content": (
                "Psoriasis is a chronic autoimmune condition that causes rapid buildup of skin cells, "
                "leading to scaling on the skin's surface. Common signs include red patches covered "
                "with thick silvery scales, dry cracked skin that may bleed, itching, burning, or "
                "soreness. Psoriasis is not contagious. Triggers include stress, skin injuries, "
                "infections, and certain medications. Treatment options include topical treatments "
                "(corticosteroids, vitamin D analogues), phototherapy, and systemic medications "
                "(biologics, methotrexate). While not a cancer, psoriasis significantly impacts "
                "quality of life."
            ),
            "metadata": {"source": "NIH/NIAMS", "topic": "Psoriasis", "category": "chronic"},
        },
        # ── Eczema / Dermatitis ─────────────────────────────────────
        {
            "content": (
                "Eczema (atopic dermatitis) is a condition that causes the skin to become inflamed, "
                "itchy, cracked, and rough. It is common in children but can occur at any age. "
                "Eczema is not contagious. Risk factors include family history of allergies, asthma, "
                "or hay fever. Common triggers include irritants, allergens, stress, and weather. "
                "Treatment focuses on managing symptoms: moisturizers, topical corticosteroids, "
                "immunosuppressants, and avoiding triggers. Eczema is a benign condition but can "
                "affect quality of life significantly due to chronic itching."
            ),
            "metadata": {"source": "NIH/NIAID", "topic": "Eczema", "category": "chronic"},
        },
        # ── When to See a Dermatologist ─────────────────────────────
        {
            "content": (
                "You should see a dermatologist immediately if you notice: a new mole or growth "
                "that is changing, a sore that does not heal within 3 weeks, a mole that is asymmetric "
                "or has irregular borders, a lesion with multiple colors, any skin spot that is "
                "bleeding without injury, rapid changes in any existing skin lesion, a dark streak "
                "under a fingernail or toenail, or a family history of melanoma combined with "
                "atypical moles. Early detection of skin cancer dramatically improves survival rates."
            ),
            "metadata": {"source": "American Academy of Dermatology", "topic": "When to See a Doctor", "category": "guidance"},
        },
    ]

    return documents

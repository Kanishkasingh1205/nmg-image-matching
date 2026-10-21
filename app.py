
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
from pathlib import Path
from threading import Thread

from match import match_images


# App settings
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

BG = "#F8FAFC"
WHITE = "#FFFFFF"
BORDER = "#E2E8F0"
TEXT = "#0F172A"
MUTED = "#64748B"

root = ctk.CTk()
root.title("NMG Image Matching System")
root.geometry("1100x850")
root.minsize(850, 650)
root.configure(fg_color=BG)

reference_path = None
query_path = None

image_a_preview = None
image_b_preview = None
matching_preview = None


# Header
header = ctk.CTkFrame(
    root, fg_color="transparent"
)
header.pack(fill="x", padx=28, pady=(25, 10))

ctk.CTkLabel(
    header,
    text="◎ NMG Image Matching System",
    font=ctk.CTkFont(size=26, weight="bold"),
    text_color=TEXT
).pack(anchor="w")

ctk.CTkLabel(
    header,
    text="Physical Object Verification using SIFT and RANSAC",
    font=ctk.CTkFont(size=13),
    text_color=MUTED
).pack(anchor="w", pady=(5, 0))


# Main scroll area
content = ctk.CTkScrollableFrame(
    root,
    fg_color="transparent"
)
content.pack(fill="both", expand=True, padx=25, pady=10)


# Image selection cards
image_row = ctk.CTkFrame(
    content, fg_color="transparent"
)
image_row.pack(fill="x", pady=(5, 15))

image_row.grid_columnconfigure(0, weight=1)
image_row.grid_columnconfigure(1, weight=1)


def create_image_card(parent, column, title):
    card = ctk.CTkFrame(
        parent,
        fg_color=WHITE,
        corner_radius=14,
        border_width=1,
        border_color=BORDER,
        height=250
    )
    card.grid(
        row=0,
        column=column,
        sticky="nsew",
        padx=7
    )
    card.grid_propagate(False)

    ctk.CTkLabel(
        card,
        text=title,
        font=ctk.CTkFont(
            size=16, weight="bold"
        ),
        text_color=TEXT
    ).pack(pady=(15, 5))

    preview = ctk.CTkLabel(
        card,
        text="No image selected",
        text_color=MUTED,
        width=360,
        height=145
    )
    preview.pack(pady=5, expand=True)

    button = ctk.CTkButton(
        card,
        text="Choose Image",
        fg_color="#EEF2FF",
        hover_color="#DBEAFE",
        text_color="#1D4ED8",
        width=180,
        height=32
    )
    button.pack(pady=(5, 15))

    return preview, button


preview_a, button_a = create_image_card(
    image_row, 0, "Reference Image"
)
preview_b, button_b = create_image_card(
    image_row, 1, "Query Image"
)


def display_image(path, label, side):
    global image_a_preview, image_b_preview

    image = Image.open(path).convert("RGB")
    image.thumbnail((390, 140))

    photo = ctk.CTkImage(
        light_image=image,
        dark_image=image,
        size=image.size
    )

    label.configure(image=photo, text="")

    if side == "a":
        image_a_preview = photo
    else:
        image_b_preview = photo


def select_reference():
    global reference_path

    path = filedialog.askopenfilename(
        title="Choose Reference Image",
        initialdir=str(
            Path("data/images").resolve()
        ),
        filetypes=[
            ("Images", "*.jpg *.jpeg *.png")
        ]
    )

    if path:
        reference_path = path
        display_image(path, preview_a, "a")


def select_query():
    global query_path

    path = filedialog.askopenfilename(
        title="Choose Query Image",
        initialdir=str(
            Path("data/images").resolve()
        ),
        filetypes=[
            ("Images", "*.jpg *.jpeg *.png")
        ]
    )

    if path:
        query_path = path
        display_image(path, preview_b, "b")


button_a.configure(command=select_reference)
button_b.configure(command=select_query)


# Compare button
compare_button = ctk.CTkButton(
    content,
    text="▶  Compare Images",
    height=48,
    corner_radius=12,
    fg_color="#2563EB",
    hover_color="#1D4ED8",
    font=ctk.CTkFont(
        size=17, weight="bold"
    )
)
compare_button.pack(
    fill="x",
    padx=7,
    pady=(5, 20)
)


# Result card
result_card = ctk.CTkFrame(
    content,
    fg_color=WHITE,
    corner_radius=14,
    border_width=1,
    border_color=BORDER
)
result_card.pack(fill="x", padx=7)

ctk.CTkLabel(
    result_card,
    text="MATCHING RESULT",
    text_color=MUTED,
    font=ctk.CTkFont(size=13)
).pack(anchor="w", padx=22, pady=(20, 5))

result_label = ctk.CTkLabel(
    result_card,
    text="Select two images to begin",
    font=ctk.CTkFont(
        size=25, weight="bold"
    ),
    text_color=TEXT
)
result_label.pack(
    anchor="w", padx=22, pady=5
)

details_label = ctk.CTkLabel(
    result_card,
    text="Results will appear after comparison.",
    font=ctk.CTkFont(size=14),
    text_color=MUTED,
    justify="left"
)
details_label.pack(
    anchor="w",
    padx=22,
    pady=(5, 22)
)


# Visualization card
visual_card = ctk.CTkFrame(
    content,
    fg_color=WHITE,
    corner_radius=14,
    border_width=1,
    border_color=BORDER
)
visual_card.pack(
    fill="x",
    padx=7,
    pady=(18, 20)
)

ctk.CTkLabel(
    visual_card,
    text="RANSAC Inlier Match Visualization",
    font=ctk.CTkFont(
        size=17, weight="bold"
    ),
    text_color=TEXT
).pack(pady=(18, 10))

visual_label = ctk.CTkLabel(
    visual_card,
    text="Matched image visualization will appear here",
    text_color=MUTED,
    height=190
)
visual_label.pack(padx=15, pady=(5, 20))


def show_results(result):
    global matching_preview

    band = result["band"]
    confidence = result["confidence"]

    if band == "SAME":
        color = "#16A34A"
    elif band == "SIMILAR":
        color = "#D97706"
    else:
        color = "#DC2626"

    result_label.configure(
        text=f"{band} — {confidence:.2f}%",
        text_color=color
    )

    details_label.configure(
        text=(
            f"Keypoints A: {result['keypoints_a']}      "
            f"Keypoints B: {result['keypoints_b']}\n\n"
            f"Total Matches: {result['total_matches']}      "
            f"Good Matches: {result['good_matches']}\n\n"
            f"RANSAC Inliers: {result['inliers']}      "
            f"Inlier Ratio: {result['inlier_ratio']:.4f}\n\n"
            f"Processing Time: {result['time_ms']:.2f} ms"
        )
    )

    output_path = Path("output") / (
        f"{Path(reference_path).stem}__"
        f"{Path(query_path).stem}.jpg"
    )

    if output_path.exists():
        image = Image.open(output_path).convert("RGB")
        image.thumbnail((960, 330))

        matching_preview = ctk.CTkImage(
            light_image=image,
            dark_image=image,
            size=image.size
        )

        visual_label.configure(
            image=matching_preview,
            text="",
            height=image.height
        )

    compare_button.configure(
        text="▶  Compare Images",
        state="normal"
    )


def show_error(error):
    compare_button.configure(
        text="▶  Compare Images",
        state="normal"
    )
    result_label.configure(
        text="Comparison failed",
        text_color="#DC2626"
    )
    messagebox.showerror("Matching Error", str(error))


def compare():
    if not reference_path or not query_path:
        messagebox.showwarning(
            "Missing Images",
            "Please choose both images first."
        )
        return

    compare_button.configure(
        text="Processing...",
        state="disabled"
    )

    result_label.configure(
        text="Matching images...",
        text_color="#2563EB"
    )

    def worker():
        try:
            result = match_images(
                reference_path,
                query_path,
                "output"
            )
            root.after(
                0, lambda: show_results(result)
            )

        except Exception as error:
            root.after(
                0,
                lambda e=str(error): show_error(e)
            )

    Thread(
        target=worker,
        daemon=True
    ).start()


compare_button.configure(command=compare)

root.mainloop()

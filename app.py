
import os
import json
import re
import tempfile
from pathlib import Path
from datetime import date, datetime

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "students.json"

START_DATE = date(2026, 10, 8)
END_DATE = date(2026, 10, 14)
TOTAL_DAYS = (END_DATE - START_DATE).days + 1

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv(
    "HF_MODEL", "Qwen/Qwen2.5-3B-Instruct"
)

client = None
if HF_TOKEN:
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
        timeout=30
    )


# ---------- JSON FILE OPERATIONS ----------

def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"JSON file not found: {DATA_FILE}\n"
            "Create students.json in the same folder as app.py."
        )

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data.get("students"), list):
        raise ValueError("JSON must contain a students list.")

    for student in data["students"]:
        student.setdefault("attendance", {})

    return data


def save_data(data):
    """Write to a temporary file, then replace the JSON file."""
    fd, temp_path = tempfile.mkstemp(
        dir=str(DATA_FILE.parent),
        prefix="attendance_",
        suffix=".tmp"
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_path, DATA_FILE)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ---------- ATTENDANCE CALCULATIONS ----------

def calculate_student(student):
    attendance = student["attendance"]

    # Only the seven dates in the project period count.
    days_present = sum(
        attendance.get((START_DATE.fromordinal(
            START_DATE.toordinal() + i
        )).isoformat()) == "present"
        for i in range(TOTAL_DAYS)
    )

    student["days_present"] = days_present
    student["total_days"] = TOTAL_DAYS
    student["attendance_percentage"] = round(
        days_present / TOTAL_DAYS * 100, 2
    )


def calculate_class_average(data):
    students = data["students"]

    if not students:
        data["class_attendance_average"] = 0.0
        return 0.0

    for student in students:
        calculate_student(student)

    average = sum(
        s["attendance_percentage"] for s in students
    ) / len(students)

    data["class_attendance_average"] = round(average, 2)
    return data["class_attendance_average"]


# ---------- COMMAND PARSING ----------

def parse_date(text):
    """Accept common written dates and ISO dates."""
    cleaned = re.sub(
        r"\b(\d{1,2})(st|nd|rd|th)\b",
        r"\1",
        text.lower()
    )
    cleaned = re.sub(r",", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    formats = [
        "%Y-%m-%d",
        "%d %B %Y",
        "%d %b %Y",
        "%B %d %Y",
        "%b %d %Y"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            pass

    return None


def local_parse(command):
    """
    Parse straightforward attendance commands without an API call.
    This ensures ordinary attendance changes do not depend on AI uptime.
    """
    text = command.lower().strip()

    roll_match = re.search(
        r"\broll(?:\s*(?:number|no\.?))?\s*[:#]?\s*"
        r"([a-z0-9_-]+)\b",
        text
    )

    # Also accept: mark 101 present ...
    if not roll_match:
        roll_match = re.search(
            r"\b(?:mark|set|show|calculate|remove|delete|get)\s+"
            r"(?:attendance\s+for\s+)?([a-z0-9_-]+)\b",
            text
        )

    date_match = re.search(
        r"\b\d{4}-\d{1,2}-\d{1,2}\b|"
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+"
        r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|"
        r"apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
        r"aug(?:ust)?|sep(?:t(?:ember)?)?|"
        r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"\s*,?\s*\d{4}\b|"
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|"
        r"apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
        r"aug(?:ust)?|sep(?:t(?:ember)?)?|"
        r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"\s+\d{1,2}(?:st|nd|rd|th)?"
        r"\s*,?\s*\d{4}\b",
        text
    )

    # Handle class-wide requests before student-specific actions.
    if "class average" in text or "class attendance average" in text:
        return {"action": "class_average"}

    if "show all" in text or "all students" in text:
        return {"action": "show_all"}

    if "help" == text:
        return {"action": "help"}

    if not roll_match:
        return None

    roll_no = roll_match.group(1)
    student_action_text = text

    if any(word in student_action_text for word in
           ["remove", "delete", "undo", "clear"]):
        action = "remove_attendance"
    elif any(word in student_action_text for word in
             ["percentage", "percent", "attendance rate"]):
        action = "percentage"
    elif "show" in student_action_text or "view" in student_action_text:
        action = "show_student"
    elif "present" in student_action_text:
        action = "mark_present"
    elif "absent" in student_action_text:
        action = "mark_absent"
    else:
        return None

    if action in {
        "mark_present", "mark_absent", "remove_attendance"
    }:
        if not date_match:
            return None

        selected_date = parse_date(date_match.group(0))
        if selected_date is None:
            return None

        return {
            "action": action,
            "roll_no": roll_no,
            "date": selected_date.isoformat()
        }

    return {"action": action, "roll_no": roll_no}


def ai_parse(command, data):
    """Use Hugging Face when local parsing cannot understand the command."""
    if client is None:
        raise RuntimeError(
            "Could not understand the command locally and "
            "HF_TOKEN is not configured."
        )

    students = [
        {"name": s["name"], "roll_no": str(s["roll_no"])}
        for s in data["students"]
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=150,
        messages=[
            {
                "role": "system",
                "content": (
                    "Interpret attendance commands. Return only a JSON object "
                    'with keys action, roll_no, date. Allowed actions: '
                    "mark_present, mark_absent, remove_attendance, "
                    "percentage, class_average, show_student, show_all, help. "
                    "Dates must use YYYY-MM-DD. For unclear commands use help. "
                    "Available students: " + json.dumps(students)
                )
            },
            {"role": "user", "content": command}
        ]
    )

    content = response.choices[0].message.content or ""
    content = re.sub(
        r"^\s*```(?:json)?\s*|\s*```\s*$",
        "",
        content.strip(),
        flags=re.IGNORECASE
    )
    result = json.loads(content)

    if not isinstance(result, dict):
        raise ValueError("The AI response was not a JSON object.")

    return result


# ---------- RESPONSE AND COMMAND EXECUTION ----------

def respond(status, message, **details):
    print(json.dumps({
        "status": status,
        "message": message,
        **details
    }, indent=2, ensure_ascii=False))


def process_command(command):
    # Always load the latest saved file for each new command.
    data = load_data()

    # Parse simple commands locally; use the HF model otherwise.
    instruction = local_parse(command)

    if instruction is None:
        try:
            instruction = ai_parse(command, data)
        except Exception as exc:
            respond(
                "error",
                "Could not interpret the command. Use a clear command "
                "or check the Hugging Face token/model.",
                error=str(exc)
            )
            return

    action = instruction.get("action")
    valid_actions = {
        "mark_present", "mark_absent", "remove_attendance",
        "percentage", "class_average", "show_student",
        "show_all", "help"
    }

    if action not in valid_actions:
        respond("error", "Invalid action returned by the parser.")
        return

    if action == "help":
        respond("success", "Examples of supported commands.", examples=[
            "Mark roll 101 present on 8 October 2026",
            "Mark roll 101 absent on 9 October 2026",
            "Remove attendance for roll 101 on 8 October 2026",
            "Calculate percentage for roll 101",
            "Calculate class average",
            "Show all students"
        ])
        return

    if action == "class_average":
        average = calculate_class_average(data)
        save_data(data)
        respond(
            "success", "Class average recalculated and saved.",
            class_attendance_average=average,
            student_count=len(data["students"]),
            saved_file=str(DATA_FILE)
        )
        return

    if action == "show_all":
        calculate_class_average(data)
        save_data(data)
        respond(
            "success", "Current student records.",
            students=data["students"],
            class_attendance_average=data["class_attendance_average"],
            saved_file=str(DATA_FILE)
        )
        return

    roll_no = str(instruction.get("roll_no", "")).strip()
    student = next(
        (s for s in data["students"]
         if str(s.get("roll_no", "")) == roll_no),
        None
    )

    if student is None:
        respond("error", "Student not found.", roll_no=roll_no)
        return

    if action in {
        "mark_present", "mark_absent", "remove_attendance"
    }:
        raw_date = str(instruction.get("date", "")).strip()

        try:
            selected_date = date.fromisoformat(raw_date)
        except ValueError:
            respond(
                "error", "Invalid date. Use YYYY-MM-DD.",
                received_date=raw_date
            )
            return

        if not START_DATE <= selected_date <= END_DATE:
            respond(
                "error", "Date is outside the attendance period.",
                allowed_start=str(START_DATE),
                allowed_end=str(END_DATE)
            )
            return

        date_key = selected_date.isoformat()

        if action == "mark_present":
            student["attendance"][date_key] = "present"
            message = "Attendance marked present and saved."

        elif action == "mark_absent":
            student["attendance"][date_key] = "absent"
            message = "Attendance marked absent and saved."

        else:
            # Remove the entry rather than writing "not_marked".
            student["attendance"].pop(date_key, None)
            message = "Attendance entry removed and saved."

        # Save attendance before reporting success.
        calculate_student(student)
        calculate_class_average(data)
        save_data(data)

        # Reload from disk so the response confirms persisted data.
        saved_data = load_data()
        saved_student = next(
            s for s in saved_data["students"]
            if str(s["roll_no"]) == roll_no
        )

        respond(
            "success",
            message,
            student=saved_student,
            class_attendance_average=saved_data[
                "class_attendance_average"
            ],
            saved_file=str(DATA_FILE)
        )
        return

    if action == "percentage":
        calculate_student(student)
        respond(
            "success", "Attendance percentage calculated.",
            roll_no=roll_no,
            name=student["name"],
            days_present=student["days_present"],
            total_days=student["total_days"],
            attendance_percentage=student["attendance_percentage"]
        )
        return

    if action == "show_student":
        calculate_student(student)
        respond("success", "Student record.", student=student)
        return


def main():
    print("ATTENDANCE MANAGEMENT SYSTEM")
    print(f"JSON file: {DATA_FILE}")
    print("Attendance period: 2026-10-08 to 2026-10-14")
    print("Type help for examples, or exit to quit.\n")

    while True:
        try:
            command = input("Enter command: ").strip()

            if command.lower() in {"exit", "quit"}:
                print('{"status":"success","message":"Goodbye."}')
                break

            if command:
                process_command(command)

        except (OSError, ValueError, KeyError) as exc:
            respond("error", "Could not process attendance data.",
                    error=str(exc))
        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
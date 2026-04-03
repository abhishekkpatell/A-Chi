import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_flow():
    try:
        print("--- 1. Resetting Environment ---")
        reset_resp = requests.post(f"{BASE_URL}/reset", json={"task_id": "login_lockout"})
        reset_resp.raise_for_status()
        obs = reset_resp.json()["observation"]
        print(f"Task: {obs['title']}")

        print("\n--- 2. Submitting Action ---")
        # Creating a policy-compliant action for the login_lockout task
        action = {
            "action_type": "finalize",
            "issue_type": "account_access",
            "priority": "medium",
            "team": "account_access",
            "status": "responded",
            "tags": ["password_reset", "login_issue"],
            "message": "Sorry for the trouble. Please check your spam folder for the reset link, try the reset flow again, and confirm 2fa on the current device if it still blocks access.",
            "internal_note": "Triaged by smoke test",
            "confidence": 1.0
        }
        step_resp = requests.post(f"{BASE_URL}/step", json=action)
        step_resp.raise_for_status()
        result = step_resp.json()
        print(f"Reward: {result['reward']['total']}")
        print(f"Current Score: {result['observation']['current_score']}")
        print(f"Feedback: {result['observation']['last_feedback'][:100]}...")
        print(f"Done: {result['done']}")
        
    except Exception as e:
        print(f"Error connecting to server: {e}")
        print("Make sure the uvicorn server is running on port 8000!")

if __name__ == "__main__":
    test_flow()

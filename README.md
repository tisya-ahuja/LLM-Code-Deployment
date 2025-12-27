# LLM Code Deployment

## Overview

This project demonstrates an **LLM-assisted build, deployment, and revision system** for web applications.  
It receives a structured JSON request describing an application, generates the app using an LLM, deploys it to **GitHub Pages**, and reports deployment details to an evaluation API.

The system supports **multiple evaluation rounds**, allowing instructor feedback to be incorporated into revised deployments.

---

## Project Workflow

The project follows a **Build → Evaluate → Revise** lifecycle.

### Build
- Accepts a JSON POST request containing an application brief
- Verifies a shared secret
- Parses request metadata and attachments
- Uses an LLM to generate a minimal working web app
- Creates a public GitHub repository
- Adds an MIT License
- Pushes code and enables GitHub Pages
- Sends repository and deployment metadata to the evaluation API

### Evaluate (Instructor Side)
- Automated static analysis
- Dynamic testing (Playwright)
- LLM-based evaluation
- Results are stored and published after the deadline
- A follow-up revision request is issued

### Revise
- Verifies the second request and secret
- Updates the application based on feedback
- Re-deploys GitHub Pages
- Sends updated repo and commit information to the evaluation API

---

## API Usage
gemini api key
  
### Endpoint
```bash
https://Tisya0-LLM-Code-Deployment.hf.space/solve
```
```bash
POST /api-endpoint
Content-Type: application/json

### Example Request
```bash
{
  "email": "student@example.com",
  "secret": "shared-secret",
  "task": "captcha-solver-123",
  "round": 1,
  "nonce": "ab12-xyz",
  "brief": "Create a captcha solver that handles ?url=https://.../image.png.",
  "checks": [
    "Repo has MIT license",
    "README.md is professional",
    "Page displays captcha URL passed at ?url=...",
    "Page displays solved captcha text within 15 seconds"
  ],
  "evaluation_url": "https://example.com/notify",
  "attachments": [
    {
      "name": "sample.png",
      "url": "data:image/png;base64,iVBORw..."
    }
  ]
}
```

## Response
- Returns HTTP 200 on success
- Retries failed evaluation submissions using exponential backoff

## Deployment Details
- Repositories are created dynamically using the GitHub API
- Repository names are derived from the task ID
- All repositories are:
-- Public
-- Licensed under MIT
-- Hosted on GitHub Pages
-- GitHub Pages must be reachable with HTTP 200 OK

## License

This project is licensed under the MIT License.
See the LICENSE file for more information.

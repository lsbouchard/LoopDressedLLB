# Release Checklist

Use this checklist before pushing the repository to GitHub.

1. Confirm the release source hash:

   ```bash
   shasum -a 256 src/jeener_packet_gaussian/simulator.py
   ```

   Expected:

   ```text
   4892b31a6bb43573c0c5051fbe5f587aa3caef14966adf51a26fc6be889c5722
   ```

2. Run the tests:

   ```bash
   python -m unittest discover -s tests
   ```

3. Run a command-line execution test:

   ```bash
   python -m jeener_packet_gaussian --help
   ```

4. Review the license.

   `LICENSE` permits non-commercial research, educational, and scholarly use
   only. It disclaims warranties and states that no technical support is
   provided.

5. Update `CITATION.cff`.

   Add the PRA DOI if desired.

6. Create a clean initial Git history:

   ```bash
   git init
   git add .
   git commit -m "Freeze Jeener packet Gaussian codebase"
   ```

7. Push to GitHub:

   ```bash
   git branch -M main
   git remote add origin git@github.com:<OWNER>/jeener-packet-gaussian.git
   git push -u origin main
   ```

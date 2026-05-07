# Deployment — Vercel

The repo ships static HTML in `reports/`. Vercel serves it directly with no build step.

## One-time setup

1. Push the repo to GitHub (`mjdup77/manchester-united-fc`).
2. Sign in at [vercel.com](https://vercel.com) with the same GitHub account.
3. **Add New… → Project → Import Git Repository → `mjdup77/manchester-united-fc`.**
4. **Framework preset:** *Other*.
5. **Root directory:** leave at repo root (the `vercel.json` at the root handles routing).
6. **Output directory:** `reports`.
7. **Build command:** *(leave empty — no build step required.)*
8. Click **Deploy**.

Vercel will assign a default URL like `https://manchester-united-fc-<hash>.vercel.app/`. To get the clean URL `https://manchester-united-fc.vercel.app/leicester-1516`:

- Settings → Domains → Add → `manchester-united-fc.vercel.app` (uses the project name as a subdomain on the free tier).

## Re-deploy

Every push to `main` triggers an automatic re-deploy. To force a re-deploy without a code change:

```bash
vercel --prod
```

(or click *Redeploy* on the project's Deployments page).

## Local sanity-check before deploy

```bash
uv run python -m analysis
python3 -m http.server -d reports 8000
# Browse to http://localhost:8000/leicester-1516.html
```

## What `vercel.json` does

```json
{
  "cleanUrls": true,
  "trailingSlash": false,
  "headers": [...]
}
```

- `cleanUrls: true` — `/leicester-1516` resolves to `/leicester-1516.html`.
- `trailingSlash: false` — canonical URL is the no-slash form.
- Three security headers (`X-Content-Type-Options`, `Referrer-Policy`, `Permissions-Policy`) — copy-paste from the personal blog template; harmless and good hygiene.

## Gotcha: `.gitignore` exception for `reports/assets/`

The figure PNGs in `reports/assets/` MUST be committed for Vercel to serve them. The `.gitignore` rule

```
reports/*.pdf
!reports/assets/
```

ignores PDFs we generate locally (Cmd-P → Save as PDF) but explicitly *un-ignores* the assets directory. Verify after any `.gitignore` edit:

```bash
git check-ignore -v reports/assets/fig1_corner_delivery_heatmap.png
# (no output means the file is NOT ignored — exactly what we want)
```

# Task: Fix Historical Prediction Trajectory timestamp and as-of data logic

## 1. Project context

The application is deployed as a Hugging Face Docker Space using FastAPI + React.

The strategy page and online strategy-account subsystem are legacy/incomplete and are not used. Simulations are run offline. GitHub Actions are not used.

Do not modify the following unless required to keep tests passing:

* strategy account persistence
* paper-trading state
* GitHub Actions workflows
* strategy scheduler behaviour
* model cycle frequency

This task is specifically about the **Historical Prediction Trajectory chart**, Layer A weather timestamps, and correct point-in-time data availability.

---

## 2. Current incorrect behaviour

The chart sometimes shows temperature or prediction data at future times, or appears to use data from the previous date.

Two separate issues are involved.

### Issue A: HKT timestamp interpreted as UTC

HKO observation timestamps are sometimes naive Hong Kong local times.

Example:

* Intended observation time: `2026-07-23 19:00 HKT`
* Incorrect stored value: `2026-07-23T19:00:00+00:00`
* Frontend displays it as: `2026-07-24 03:00 HKT`

Likely location:

* `app/services/canonical_cycle.py`
* Search for `_parse_datetime(...)`
* Search for `naive_timezone=timezone.utc`
* Source value comes from `weather_service.get_intraday_state()["time_now"]`

A naive HKO wall-clock timestamp must be localized as `Asia/Hong_Kong`, then converted to UTC for storage.

### Issue B: Observation time is confused with release/capture time

HKO may release delayed observations.

Example:

* Current system time / first received: `09:48`
* HKO data covers observations up to: `09:40`
* Actual temperature chart must stop at `09:40`
* A new model prediction may exist at `09:48`

The chart must not relabel the `09:40` observation as an `09:48` actual temperature point.

---

## 3. Required timestamp semantics

Use explicit meanings consistently.

### `observation_timestamp`

The time represented by the weather observation.

Use this as the x-axis for actual temperature.

Example: `09:40`.

### `source_release_timestamp`

The release timestamp supplied by HKO, if available.

May be null.

### `first_seen_timestamp`

The earliest time our system successfully obtained this observation or correction.

If HKO does not provide a release timestamp, this is the safe availability timestamp for simulation.

Example: `09:48`.

### `capture_timestamp`

When the current collector request completed.

This is operational metadata, not the actual-temperature x-axis.

### `decision_timestamp`

When the model cycle made its prediction.

Use this as the x-axis for prediction trajectories.

Example: `09:48`.

---

## 4. Required fixes

### P0-1: Correct naive HKT parsing

Inspect `app/services/canonical_cycle.py`.

Where weather observation timestamps from HKO are parsed, do not interpret naive timestamps as UTC.

Use `ZoneInfo("Asia/Hong_Kong")` or the existing shared HKT timezone constant.

Required conversion:

* Input: naive `2026-07-23 19:00`
* Interpret as: `2026-07-23 19:00+08:00`
* Store canonical UTC: `2026-07-23T11:00:00+00:00`

Do not change timestamps that already contain an explicit timezone offset.

Add a regression test.

---

### P0-2: Preserve observation time and availability time separately

Inspect the weather collector, canonical-cycle linkage, Layer A weather schema and serialization.

Ensure each new weather observation stores:

* `observation_timestamp`
* `first_seen_timestamp`
* `capture_timestamp`
* `source_release_timestamp`, if available

For backward compatibility, `snapshot_timestamp` may remain, but define it consistently as the observation timestamp or deprecate it explicitly.

Do not use one field to mean different things in different collectors.

When the same observation is fetched repeatedly:

* preserve the earliest `first_seen_timestamp`
* do not replace it with a later polling time

When a corrected HKO value appears later:

* preserve it as a later version/correction
* its correction must only become available from its own first-seen time
* do not make a later correction visible to an earlier model cycle

---

### P0-3: Fix actual-temperature chart construction

Inspect:

* `layer_a/minute_view.py`
* chart/history API routers
* historical-store projection logic

Actual-temperature points must be keyed by:

`observation_timestamp`

They must not be keyed by:

* `capture_timestamp`
* API request time
* current chart refresh time
* model decision time

Do not forward-fill raw actual-temperature observations to every minute.

At `09:48`, when the latest released observation is `09:40`:

* actual-temperature series ends at `09:40`
* no actual-temperature points exist at `09:41`–`09:48`
* prediction series may contain a `09:48` model point

Feature engineering may internally forward-fill a value, but raw chart points must remain at their true observation times.

---

### P0-4: Enforce point-in-time availability for simulation/replay

At decision time `t`, an observation is eligible only when:

`first_seen_timestamp <= t`

Do not use only:

`observation_timestamp <= t`

That would introduce look-ahead leakage.

Example:

| Decision time | 09:40 observation first seen at 09:48 | Eligible |
| ------------- | ------------------------------------- | -------- |
| 09:45         | No                                    | No       |
| 09:47         | No                                    | No       |
| 09:48         | Yes                                   | Yes      |
| 09:50         | Yes                                   | Yes      |

Selection logic should conceptually be:

1. Filter weather versions where `first_seen_timestamp <= decision_timestamp`.
2. Apply corrections only after their own first-seen timestamps.
3. Select the latest eligible observation by `observation_timestamp`.
4. Save the selected observation ID and timestamps into the model-cycle lineage.

Every model-cycle output should expose:

* `decision_timestamp`
* `weather_data_through`
* `weather_first_seen_timestamp`
* `weather_age_seconds`
* selected weather snapshot/version ID

---

### P0-5: Do not generate future chart rows

For a selected HKT date:

* past date: allow rows only within that calendar date
* today: allow rows only up to current HKT time
* future date: return no trajectory rows

However, actual temperature may naturally stop before current time because the latest HKO observation is delayed.

Example at current time `09:48`:

* chart maximum allowed time: `09:48`
* actual-temperature final point: possibly `09:40`
* model final point: possibly `09:48`

Filter by the HKT calendar date, not by the UTC calendar date.

No row selected for `2026-07-23` may display as `2026-07-24` in Hong Kong time.

---

### P0-6: Fix model trajectory rendering without synthesizing model cycles

Do not add an arbitrary hard-coded 15-minute model-cycle limit as the main fix.

Do not change the configured model inference cadence in this task.

A model point must exist only when a real model cycle was executed.

Use:

`x = decision_timestamp`

Do not create fake per-minute model cycles by copying the latest prediction into generated rows.

For visual continuity, the frontend may render a step line between actual model-cycle points. It must not create new observations or imply that inference ran every minute.

After the final real cycle:

* either stop the line at the final cycle; or
* optionally extend visually to the current time while marking it as carried/stale

Any stale threshold must be derived from configured expected cycle interval, not hard-coded independently.

Examples:

* expected cycle = 5 minutes → warning may begin after 10–15 minutes
* expected cycle = 1 minute → warning may begin after 2–3 minutes

Expose:

* actual model-cycle timestamp
* model age
* stale status

Do not treat carried-forward values as new entry opportunities in simulation.

---

### P0-7: Frontend must explicitly display Hong Kong time

Inspect the React chart component, likely `ModelsComparisonChart.tsx` or equivalent.

All date formatting must specify:

`timeZone: "Asia/Hong_Kong"`

Do not rely on the browser timezone.

Axis labels may show `HH:mm`, but tooltip must show the complete local date and time, for example:

`23 Jul 2026, 09:48`

Tooltip for a prediction point should show:

* prediction time
* latest weather observation time
* weather first-seen time
* weather age
* model-cycle ID

Tooltip for an actual-temperature point should show:

* observation time
* first-seen time
* release lag

Avoid connecting invalid or cross-date rows using `connectNulls`.

---

### P0-8: Safely handle already-corrupted historical records

Do not mutate immutable raw Layer A files in place.

At read/projection time, reject records where:

`observation_timestamp > capture_timestamp + allowed_clock_skew`

Use a small tolerance such as five minutes.

Known corrupted records from `canonical_cycle_link` may be approximately eight hours in the future because naive HKT was interpreted as UTC.

For the first patch:

* exclude these corrupted records from the chart
* count and report them in quality metadata
* do not silently auto-correct them unless the repair rule is deterministic and fully tested

Return diagnostics such as:

* `excluded_future_weather_records`
* `duplicate_observation_versions`
* `latest_weather_observation_timestamp`
* `latest_weather_first_seen_timestamp`

---

### P0-9: Deduplicate chart observations without breaking as-of simulation

Layer A contains duplicate weather snapshot IDs and repeated observations.

Do not simply drop duplicates globally.

Maintain version-aware records for replay.

For the observational chart:

* one point per observation timestamp/source/station identity
* prefer a valid, non-fallback, latest corrected value
* expose correction metadata where relevant

For model replay:

* use only versions available by the model decision time
* a later correction must not leak into an earlier cycle
* preserve earliest first-seen timestamp for the original observation
* preserve separate first-seen timestamp for later corrections

Do not use the final corrected historical value for all earlier model cycles.

---

## 5. Required regression tests

Add focused tests with small synthetic records.

### Test 1: Naive HKT conversion

Input:

`2026-07-23 19:00` with no timezone

Expected canonical UTC:

`2026-07-23T11:00:00+00:00`

Expected HKT display:

`2026-07-23 19:00`

---

### Test 2: Delayed HKO release

Record:

* observation: `09:40`
* first seen: `09:48`
* capture: `09:48:10`

At chart refresh `09:48`:

* actual temperature ends at `09:40`
* no actual points from `09:41` through `09:48`
* model prediction may exist at `09:48`

---

### Test 3: Point-in-time replay

The `09:40` observation first seen at `09:48`:

* unavailable to model cycle at `09:47`
* available to model cycle at `09:48`

---

### Test 4: Cross-date protection

A record stored with an incorrect UTC interpretation must not appear on the next HKT calendar date.

Selecting `2026-07-23` must return only records whose Hong Kong local date is `2026-07-23`.

---

### Test 5: Future-corrupt record rejection

Input:

* capture: `11:09 UTC`
* observation: `19:00 UTC`
* collector: `canonical_cycle_link`

Expected:

* excluded from chart projection
* quality counter incremented

---

### Test 6: Real model cycles only

Given real model cycles at:

* `09:40`
* `09:45`
* `09:50`

The backend must not synthesize cycles at:

* `09:41`
* `09:42`
* etc.

Frontend step rendering is allowed, but API metadata must still identify the real source cycle.

---

### Test 7: Browser timezone independence

Formatting must show the same Hong Kong date/time regardless of browser or test environment timezone.

---

### Test 8: Correction availability

Original value:

* observation `09:40`
* first seen `09:48`

Corrected value:

* same observation
* correction first seen `10:05`

Expected:

* `09:50` replay uses original value
* `10:05` or later replay may use corrected value

---

## 6. Constraints

* Keep changes minimal and focused.
* Do not redesign the whole Layer A architecture.
* Do not change model cadence.
* Do not modify offline simulation strategy logic except where needed to enforce point-in-time weather availability.
* Do not modify unused strategy-account or GitHub Actions code.
* Do not rewrite raw immutable historical files.
* Prefer shared timestamp helpers over repeated local conversions.
* All stored canonical timestamps should be timezone-aware.
* Use UTC for storage and `Asia/Hong_Kong` for domain/calendar interpretation and frontend display.
* Avoid fallback behaviour that silently treats corrupt data as valid.

---

## 7. Deliverables

After implementation, report:

1. Root causes confirmed.
2. Files changed.
3. Exact timestamp semantics adopted.
4. How delayed HKO releases are represented.
5. How simulation prevents look-ahead leakage.
6. How existing corrupt records are handled.
7. Tests added and results.
8. Any remaining ambiguity in the HKO source timestamps.
9. A concise before/after example using:

   * current time `09:48`
   * latest observation `09:40`

Run the smallest relevant tests first, then the broader Layer A and API test suites.

Do not claim success only because the frontend looks correct. Verify the API payload timestamps and point-in-time replay logic.

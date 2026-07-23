import assert from "node:assert/strict"
import { execFileSync } from "node:child_process"
import test from "node:test"
import {
  buildTrajectoryTooltip,
  canConnectRealModelCycles,
  formatHktDateTime,
  formatHktTime,
  segmentCoordinatesByCadence,
} from "../src/lib/trajectoryFormatting.js"

test("HKT formatting is independent of the browser/runtime timezone", () => {
  const timestamp = "2026-07-23T01:48:00+00:00"
  assert.equal(formatHktTime(timestamp), "09:48")
  assert.equal(formatHktDateTime(timestamp), "23 Jul 2026, 09:48 HKT")

  const moduleUrl = new URL("../src/lib/trajectoryFormatting.js", import.meta.url).href
  const expression = `import(${JSON.stringify(moduleUrl)}).then(({formatHktDateTime}) => process.stdout.write(formatHktDateTime(${JSON.stringify(timestamp)})))`
  const rendered = ["UTC", "America/New_York"].map((tz) => execFileSync(
    process.execPath,
    ["--input-type=module", "--eval", expression],
    { env: { ...process.env, TZ: tz }, encoding: "utf8" },
  ))
  assert.deepEqual(rendered, ["23 Jul 2026, 09:48 HKT", "23 Jul 2026, 09:48 HKT"])
})

test("trajectory tooltip distinguishes actual observation and prediction lineage", () => {
  const tooltip = buildTrajectoryTooltip({
    timestamp: "2026-07-23T01:48:00+00:00",
    metadata: {
      actual_observation_timestamp: null,
      actual_first_seen_timestamp: null,
      actual_source_release_timestamp: "2026-07-23T01:45:00+00:00",
      actual_release_lag_seconds: 180,
      prediction_decision_timestamp: "2026-07-23T01:48:00+00:00",
      weather_data_through: "2026-07-23T01:40:00+00:00",
      weather_first_seen_timestamp: "2026-07-23T01:48:00+00:00",
      weather_age_seconds: 480,
      model_cycle_id: "cycle-0948",
      model_age_seconds: 0,
    },
    entries: [{ seriesName: "Model A", value: 30.2, color: "#fff" }],
    expectedCycleIntervalSeconds: 300,
  })

  for (const value of [
    "23 Jul 2026, 09:48 HKT",
    "Actual observation",
    "23 Jul 2026, 09:40 HKT",
    "Actual first seen",
    "Actual source release",
    "Release lag",
    "3m",
    "Prediction decision",
    "Weather data through",
    "Weather age",
    "8m",
    "Model-cycle ID",
    "cycle-0948",
  ]) assert.match(tooltip, new RegExp(value))
})

test("only real cycles within the configured cadence may be visually joined", () => {
  assert.equal(canConnectRealModelCycles("2026-07-23T09:40:00+08:00", "2026-07-23T09:50:00+08:00", 300), true)
  assert.equal(canConnectRealModelCycles("2026-07-23T09:40:00+08:00", "2026-07-23T10:01:00+08:00", 300), false)
  assert.equal(canConnectRealModelCycles("2026-07-23T23:59:00+08:00", "2026-07-24T00:04:00+08:00", 300), false)
  assert.equal(canConnectRealModelCycles("2026-07-23T09:40:00+08:00", "2026-07-23T09:45:00+08:00", null), false)
})

test("consensus coordinates are split across stale gaps", () => {
  const coordinates = [
    [Date.parse("2026-07-23T09:40:00+08:00"), 30.0],
    [Date.parse("2026-07-23T09:45:00+08:00"), 30.1],
    [Date.parse("2026-07-23T10:30:00+08:00"), 30.3],
  ]
  assert.deepEqual(segmentCoordinatesByCadence(coordinates, 300), [
    coordinates.slice(0, 2),
    coordinates.slice(2),
  ])
})

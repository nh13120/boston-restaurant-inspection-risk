# Status: data ready (Phases 0 to 2)

Pulled Boston restaurant inspections on Nov 11: 444,574 violation rows, 2016 to Nov 2026, no owner names.

Label: fail = any fail code (HE_Fail, FailExt, Hearing, closures). HE_Filed counts as a pass.

Re-inspections (a visit within 30 days of a failed one) are dropped after features are built: 40,304 of 98,907.

Model table: 58,603 routine inspections. Train 35,915 (2016 to 2022), validation 5,520 (2023), test 17,168 (2024 onward). Fail rate 59%.

Odd things: 59% is high, so random order already "hits" often, and at 20% capacity we can catch at most a third of failures. 2020 is thin. Comments held some names and emails, now masked.

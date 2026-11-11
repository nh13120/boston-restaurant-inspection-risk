# Status: data ready and audited (Phases 0 to 2, Gate 1)

Pulled Boston restaurant inspections on Nov 11: 785,315 violation rows, 2006 to Nov 2026, no owner names. Years before 2016 only feed the history features.

Label: fail = any fail code (HE_Fail, FailExt, Hearing, closures). HE_Filed counts as a pass.

Re-inspections (a visit within 30 days of a failed one) are dropped after features are built: 40,360 of 98,924.

Model table: 58,564 routine inspections. Train 35,872 (2016 to 2022), validation 5,521 (2023), test 17,171 (2024 onward). Fail rate 59%.

Odd things: 59% is high, so random order already hits often, and at 20% capacity we can catch at most a third of failures. 2020 is thin. Names and emails in comments are masked.

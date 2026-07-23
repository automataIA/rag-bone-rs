/// Throttles progress events to meaningful percentage changes. The percentage
/// describes completed work units, not elapsed or estimated remaining time.
pub struct Progress {
    total: usize,
    last_reported: Option<usize>,
}

impl Progress {
    const STEP_PERCENT: usize = 5;

    pub fn new(total: usize) -> Self {
        Self {
            total,
            last_reported: None,
        }
    }

    /// Return a percentage when callers should emit an event. Always reports
    /// the first observation and completion, with intermediate events at
    /// roughly five-percentage-point intervals.
    pub fn update(&mut self, completed: usize) -> Option<usize> {
        let percent = completed
            .min(self.total)
            .saturating_mul(100)
            .checked_div(self.total)
            .unwrap_or(100);
        let should_report = self.last_reported.is_none_or(|last| {
            percent > last && (percent == 100 || percent >= last.saturating_add(Self::STEP_PERCENT))
        });
        if should_report {
            self.last_reported = Some(percent);
            Some(percent)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_start_intervals_and_completion() {
        let mut progress = Progress::new(100);
        assert_eq!(progress.update(0), Some(0));
        assert_eq!(progress.update(4), None);
        assert_eq!(progress.update(5), Some(5));
        assert_eq!(progress.update(99), Some(99));
        assert_eq!(progress.update(100), Some(100));
        assert_eq!(progress.update(100), None);
    }

    #[test]
    fn empty_work_is_complete() {
        assert_eq!(Progress::new(0).update(0), Some(100));
    }
}

module.exports = async ({ github, context, core }) => {
	const fs = require('fs');

	let current = JSON.parse(fs.readFileSync('benchmark-results.json', 'utf8'))
	const isPush = context.eventName === 'push' && context.ref === 'refs/heads/main';
	const storeOutcome = process.env.STORE_OUTCOME || 'success';
	const THRESHOLD = 1.20;

	// Build baseline map
	const baselinePath = 'gh-pages-baseline/dev/bench/data.json';
	let baseline = {};
	let hasBaseline = false;

	if (fs.existsSync(baselinePath)) {
		try {
			const parsed = JSON.parse(fs.readFileSync(baselinePath, 'utf8'));
			const entries = parsed.entries || {};

			for (const group of Object.values(entries)) {
				if (Array.isArray(group) && group.length > 0) {
					const latest = group[group.length - 1];
					for (const b of latest.benches || []) {
						if (b.name && b.value !== undefined) {
							baseline[b.name] = b.value;
						}
					}
				}
			}

			hasBaseline = Object.keys(baseline).length > 0;
		} catch (err) {
			core.warning(`Could not read baseline data: ${err.message}`);
		}
	}

	// Build check output summary
	let hasRegressions = false;
	let summary = '';

	if (hasBaseline) {
		summary = '### Benchmark Results (compared against main)\n\n';
		summary += '| Status | Benchmark | Baseline | Current | Change |\n';
		summary += '| --- | --- | --- | --- | --- |\n';

		for (const b of benchmarks) {
			const cur = b.real_time;
			const unit = b.time_unit || 'ns';
			const bl = baseline[b.name];

			if (bl !== undefined) {
				const pct = ((cur - bl) / bl * 100).toFixed(1);
				const isRegressed = cur / bl >= THRESHOLD;
				if (isRegressed) hasRegressions = true;
				const icon = isRegressed ? '🔴' : '✅';
				summary += `| ${icon} | ${b.name} | ${bl.toFixed(2)} ${unit} | ${cur.toFixed(2)} ${unit} | ${pct}% |\n`;
			} else {
				summary += `| 🔵 | ${b.name} (new) | — | ${cur.toFixed(2)} ${unit} | — |\n`;
			}
		}
	} else {
		summary = '### Benchmark Results\n\n';
		summary += '| Benchmark | Time |\n';
		summary += '| --- | --- |\n';
		for (const b of benchmarks) {
			summary += `| ${b.name} | ${b.real_time.toFixed(2)} ${b.time_unit || 'ns'} |\n`;
		}
	}

	// Determine conclusion
	let conclusion = 'success';
	let title = 'Benchmark Results';

	if (isPush) {
		conclusion = storeOutcome === 'failure' ? 'failure' : 'success';
	} else if (hasBaseline && hasRegressions) {
		conclusion = 'failure';
		title = 'Benchmark Results — regression detected';
	} else if (!hasBaseline) {
		conclusion = 'neutral';
		title = 'Benchmark Results (no baseline)';
	}

	if (conclusion === 'failure') {
		summary += '\n---\n_Some benchmarks exceeded the 120% regression threshold._';
	}

	// Create the Check Run
	await github.rest.checks.create({
		owner: context.repo.owner,
		repo: context.repo.repo,
		name: 'Benchmark Results',
		head_sha: context.sha,
		status: 'completed',
		conclusion: conclusion,
		output: {
			title: title,
			summary: summary
		}
	});
};
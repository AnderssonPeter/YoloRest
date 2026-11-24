class ConfidenceEvaluator:
    def __init__(self, expression: str):
        self.rules = self._parse_expression(expression)

    def _parse_expression(self, expression: str) -> dict[str, tuple[float, float] | float]:
        rules: dict[str, tuple[float, float] | float] = {}
        for rule in expression.split(","):
            if ":" in rule:
                label, confidence_range = rule.split(":")
                if "-" in confidence_range:
                    min_conf, max_conf = map(float, confidence_range.split("-"))
                    rules[label] = (min_conf, max_conf)
                else:
                    rules[label] = (float(confidence_range), 1)
            else:
                # Default rule for any other label
                rules["default"] = float(rule)
        return rules

    def evaluate(self, label: str, confidence: float) -> bool:
        rule = self.rules.get(label)
        if isinstance(rule, tuple):
            min_conf, max_conf = rule
            return min_conf <= confidence <= max_conf

        default = self.rules.get("default")
        if isinstance(default, float):
            return confidence >= default
        return False

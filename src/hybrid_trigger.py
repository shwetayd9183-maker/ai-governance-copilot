# src/hybrid_trigger.py


def compute_expected_loss(current_price, predicted_drop_pct, arrival_quantity):
    """
    Estimate expected economic loss in Rs.
    Simplified approximation:
    Loss = price_drop * arrival_quantity
    """

    price_drop = current_price * predicted_drop_pct
    expected_loss = price_drop * arrival_quantity

    return expected_loss


def should_intervene(prob_severe, expected_loss, fiscal_cost, probability_threshold=0.6):
    """
    Hybrid governance trigger:
    - High probability of severe crash
    - Economic loss exceeds fiscal cost
    """

    if prob_severe >= probability_threshold and expected_loss > fiscal_cost:
        return True

    return False


def generate_trigger_decision(prob_severe, current_price, predicted_drop_pct,
                              arrival_quantity, fiscal_cost):
    """
    Full intervention decision logic.
    """

    expected_loss = compute_expected_loss(
        current_price,
        predicted_drop_pct,
        arrival_quantity
    )

    decision = should_intervene(
        prob_severe,
        expected_loss,
        fiscal_cost
    )

    return {
        "prob_severe": prob_severe,
        "expected_loss": expected_loss,
        "fiscal_cost": fiscal_cost,
        "intervene": decision
    }
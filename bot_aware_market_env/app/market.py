class AMM:
    def __init__(self, reserve_x: float = 10000.0, reserve_y: float = 10000.0):
        self.reserve_x = reserve_x
        self.reserve_y = reserve_y
        self.k = self.reserve_x * self.reserve_y

    def get_price(self):
        return self.reserve_y / self.reserve_x

    def estimate_out(self, amount_in: float, is_x_to_y: bool):
        # returns (amount_out, slippage_percentage)
        if amount_in <= 0:
            return 0.0, 0.0
            
        if is_x_to_y:
            new_r_x = self.reserve_x + amount_in
            new_r_y = self.k / new_r_x
            amount_out = self.reserve_y - new_r_y
            slippage = (amount_in / self.reserve_x) * 100  # simplified impact
        else:
            new_r_y = self.reserve_y + amount_in
            new_r_x = self.k / new_r_y
            amount_out = self.reserve_x - new_r_x
            slippage = (amount_in / self.reserve_y) * 100
            
        return amount_out, slippage

    def swap(self, amount_in: float, is_x_to_y: bool):
        amount_out, slippage_pct = self.estimate_out(amount_in, is_x_to_y)
        if is_x_to_y:
            self.reserve_x += amount_in
            self.reserve_y -= amount_out
        else:
            self.reserve_y += amount_in
            self.reserve_x -= amount_out
        return amount_out, slippage_pct

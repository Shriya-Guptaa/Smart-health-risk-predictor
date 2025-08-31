def calculate_bmi(weight_kg,height_cm):
    """ calcuates bmi from weight and height given by user 
    bmi= (weight(kg))/(height(m))^2 """
    if (height_cm <=0) or (weight_kg<=0):
        raise ValueError("Height/Weight should be greater than zero.")
    
    height_m=height_cm/100
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 2)
    
import math
def unit_conversion(conversion_direction, coef_J_to_E, hbar=1., c=1., k=1., epsilon0=1.):
    '''
    The function gives the transformation rules of units between 
    the international unit (SI) and flexible Unit (FU).
    
    Hence, all quantities in FU have the unit of energy.
    We start with the unit of \hbar and c in SI:
    \hbar = 1.05457*10^(-34) J*s
    c = 2.99792*10^8 m/s
    If we introduce a NEW quantity E, which is defined by the relation:
    #################################################################
    ----->>>>>   1 E = x J, (x is a value and J is Joules),
    #################################################################
    we can obtain the expressions of meter in terms of E via
        \hbar*c = 3.16152*10^(-26) J*m = 1
                = 3.16152*10^(-26)/x E*m = (\hbar*c),
    thus:
    #################################################################
    ----->>>>>   1 m = 3.16304*10^(25)*hbar*c*x E^(-1).
    #################################################################
    Accordingly, the expression of time in terms of E is
        c = 2.99792*10^8 m/s
          = 2.99792*10^8*(hbar*c)*3.16303*10^(25)*x E^(-1)/s,
    thus:
    #################################################################
    ----->>>>>   1 s = 9.48253*10^(33)*hbar*x E^(-1).
    #################################################################
    The mass of a particle satisfies the following ralation:
        1 J = 1/x E
            = 1 Kg*(m/s)^2 = 1.11265*10(-17) Kg
    thus:
    #################################################################
    ----->>>>>   1 Kg = 8.98752*10^(16)/(x*c**2) E.
    #################################################################
    The Boltzmann constant relates the energy and Kelvin via
        k = 1.380649*10^(-23) J/K
          = 1.380649*10^(-23)/x E/K
    thus:
    #################################################################
    ----->>>>>   1 K = 1.38065*10^(-23)/(x*k) E.
    #################################################################   
    For electromagnetic forces, the finite structure constant 
        \alpha = e^2/(4*pi*epsilon0*hbar*c) = 1/137,
    thus:
    #################################################################
    ----->>>>>   1 e = 0.302862*sqrt(hbar*c*epsilon0),
    #################################################################
    the relation between Ampere and Column is related by
        1 e = 1.602176634*10^(-19) C =0.302862*sqrt(hbar*c*epsilon0)
            = 1.602176634*10^(-19)*9.48252*10^(33)*(hbar)*x A/E
    thus:
    #################################################################
    ----->>>>>   1 A = 1.99347*10^(-16)*sqrt(c*epsilon0)/(x*sqrt(hbar)) E,
    ----->>>>>   1 C = 1.89032*10^(18)*sqrt(hbar*c*epsilon0).
    #################################################################
    The magneticfield B has the unit of Tesle in SI
        1 Tesla = 1 Kg/(A*s^2),
    thus:
    #################################################################
    ----->>>>>   1 Tesla = 5.01397*10^(-36)/(x^2*hbar^(3/2)*c^(5/2)*sqrt(epsilon0)) E^2.
    #################################################################
    The electric field has the unit of Volt/m in SI
        1 Volt/m = 1 J/(s*A*m)
    thus:
    #################################################################
    ----->>>>>   1 Volt/m = 1.67248*10^(-44)/(x^2*hbar^(3/2)*c^(3/2)*sqrt(epsilon0)) E^2.
    #################################################################
    The unit for momentum is 
    #################################################################
    ----->>>>>   1 Kg*m/s = 2.99792*10^8/(x*c) E.
    #################################################################
    The unit for force is 
    #################################################################
    ----->>>>>   1 Kg*m/s^2 = 3.16152*10^-26/(x^2*hbar*c) E^2.
    #################################################################
    
    
    params
    ======
    conversion_direction:
        two choices: 'SI_to_FU' and 'FU_to_SI'
    coef_J_to_E: 
        1 J = 1/x E, where x = coef_J_to_E
    hbar, c, epsilon0, k:
        the values chosen for converting the units 
    return
    ======
    conversion_dictionary:
        the coeficients (coef) that maps the value in SI/FU to FU/SI,
        e.g., if conversion_direction='SI_to_LHQCD', simply use
        coef*# (in SI) = # (in FU)
    '''
    
    x = coef_J_to_E
    conversion_dictionary =  \
        {'Joules':1/x,'meter': 3.16304*10**(25)*hbar*c*x,'second':9.48253*10**(33)*hbar*x,\
         'kilogram':8.98752*10**(16)/(x*c**2),'Kelvin':1.38065*10**(-23)/(x*k),\
         'Ampere':1.99347*10**(-16)*math.sqrt(c*epsilon0)/(x*math.sqrt(hbar)),\
         'Coulomb':1.89032*10**(18)*math.sqrt(hbar*c*epsilon0),\
         'Tesla':5.01397*10**(-36)/(x**2*hbar**(3/2)*c**(5/2)*math.sqrt(epsilon0)),\
         'Volt/m':1.67248*10**(-44)/(x**2*hbar**(3/2)*c**(3/2)*math.sqrt(epsilon0)), \
         'momentum':2.99792*10**8/(x*c),'force':3.16152*10**-26/(x**2*hbar*c),\
         'unit charge': 0.302862*math.sqrt(hbar*c*epsilon0)}

    # the dict of the coefficients
    if conversion_direction == 'SI_to_LHQCD':
        return conversion_dictionary
    if conversion_direction == 'LHQCD_to_SI':
        conversion_dictionaryy =  \
        {'TO_Joules':1/conversion_dictionary['Joules'],'TO_meter': 1/conversion_dictionary['meter'],\
         'TO_second':1/conversion_dictionary['second'],'TO_kilogram':1/conversion_dictionary['kilogram'],\
         'TO_Kelvin':1/conversion_dictionary['Kelvin'],'TO_Ampere':1/conversion_dictionary['Ampere'],\
         'TO_Coulomb':1/conversion_dictionary['Coulomb'],'TO_Tesla':1/conversion_dictionary['Tesla'],\
         'TO_Volt/m':1/conversion_dictionary['Volt/m'], 'TO_momentum':1/conversion_dictionary['momentum'],\
         'TO_force':1/conversion_dictionary['force'],'TO_unit charge':1/conversion_dictionary['unit charge']}
        
        return conversion_dictionaryy
    
    
def determine_coefficient_for_unit_conversion(df, dx, dp, Q, f, v, dt):
    '''Determine the proper coefficient for unit conversion between SI and LHQCD.
    '''
    
    # (df/dx)*(df/dp) ~ 1, gives the value of hbar
    hbar = 1.05457*10**(-34)*df**(2/7)/(dp*dx)**(1/7)
    
    # (df/dt)~1, gives
    x = 1.23681*10**(-136)*df/(dt*hbar**4)
    
    # (df/dx)~(df/dp), gives c*x = 3.07863*10**(-9)*math.sqrt(dp/(dx*hbar))
    c = 3.07863*10**(-9)*math.sqrt(dp/(dx*hbar))/x
    
    # Q*f*dp**3 ~ 1, gives 
    epsilon0 = 2.80258*10**116*c**5*x**6*hbar**5/(dp**6*f**2*Q**2)
    
    return hbar, c, x, epsilon0
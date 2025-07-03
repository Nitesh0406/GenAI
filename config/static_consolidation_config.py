
consolidations_day_mapping = {
    # 5-day scenarios
    'Mon_Tue_Wed_Thu_Fri': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Mon_Tue_Wed_Thu_Sat': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Tue_Wed_Thu_Sun': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': 0
    },
    'Mon_Tue_Wed_Fri_Sat': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': 0,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Tue_Wed_Fri_Sun': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': 0,
        'Sat': -1,
        'Sun': 0
    },
    'Mon_Tue_Thu_Fri_Sat': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': 0,
        'Fri': 0,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Tue_Thu_Fri_Sun': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': 0,
        'Fri': 0,
        'Sat': -1,
        'Sun': 0
    },
    'Mon_Wed_Thu_Fri_Sat': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': 0,
        'Fri': 0,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Wed_Thu_Fri_Sun': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': 0,
        'Fri': 0,
        'Sat': -1,
        'Sun': 0
    },
    'Tue_Wed_Thu_Fri_Sat': {
        'Mon': -2,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': 0,
        'Sat': 0,
        'Sun': -1
    },
    'Tue_Wed_Thu_Fri_Sun': {
        'Mon': -1,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': 0,
        'Sat': -1,
        'Sun': 0
    },

    # 4-day scenarios
    'Mon_Tue_Wed_Thu': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': -3
    },
    'Mon_Tue_Wed_Fri': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Mon_Tue_Wed_Sat': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Tue_Wed_Sun': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': -3,
        'Sun': 0
    },
    'Mon_Tue_Thu_Fri': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': 0,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Mon_Tue_Thu_Sat': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': 0,
        'Fri': -1,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Tue_Thu_Sun': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': 0
    },
    'Mon_Wed_Thu_Fri': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': 0,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Mon_Wed_Thu_Sat': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Wed_Thu_Sun': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': 0
    },
    'Tue_Wed_Thu_Fri': {
        'Mon': -3,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Tue_Wed_Thu_Sat': {
        'Mon': -2,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': 0,
        'Sun': -1
    },
    'Tue_Wed_Thu_Sun': {
        'Mon': -1,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': 0
    },

    # 3-day scenarios
    'Mon_Tue_Wed': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': -3,
        'Sun': -4
    },
    'Mon_Tue_Thu': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': -3
    },
    'Mon_Tue_Fri': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': -2,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Mon_Tue_Sat': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': -2,
        'Fri': -3,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Tue_Sun': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': -2,
        'Fri': -3,
        'Sat': -4,
        'Sun': 0
    },
    'Mon_Wed_Thu': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': -3
    },
    'Mon_Wed_Fri': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': -1,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Mon_Wed_Sat': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Wed_Sun': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': -3,
        'Sun': 0
    },
    'Tue_Wed_Thu': {
        'Mon': -4,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': -3
    },
    'Tue_Wed_Fri': {
        'Mon': -3,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Tue_Wed_Sat': {
        'Mon': -2,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': 0,
        'Sun': -1
    },
    'Tue_Wed_Sun': {
        'Mon': -1,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': -3,
        'Sun': 0
    },

    # 2-day scenarios
    'Mon_Tue': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': -2,
        'Fri': -3,
        'Sat': -4,
        'Sun': -5
    },
    'Mon_Wed': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': -3,
        'Sun': -4
    },
    'Mon_Thu': {
        'Mon': 0,
        'Tue': -1,
        'Wed': -2,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': -3
    },
    'Mon_Fri': {
        'Mon': 0,
        'Tue': -1,
        'Wed': -2,
        'Thu': -3,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Mon_Sat': {
        'Mon': 0,
        'Tue': -1,
        'Wed': -2,
        'Thu': -3,
        'Fri': -4,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Sun': {
        'Mon': 0,
        'Tue': -1,
        'Wed': -2,
        'Thu': -3,
        'Fri': -4,
        'Sat': -5,
        'Sun': 0
    },
    'Tue_Wed': {
        'Mon': -5,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': -3,
        'Sun': -4
    },
    'Tue_Thu': {
        'Mon': -4,
        'Tue': 0,
        'Wed': -1,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': -3
    },
    'Tue_Fri': {
        'Mon': -3,
        'Tue': 0,
        'Wed': -1,
        'Thu': -2,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Tue_Sat': {
        'Mon': -2,
        'Tue': 0,
        'Wed': -1,
        'Thu': -2,
        'Fri': -3,
        'Sat': 0,
        'Sun': -1
    },
    'Tue_Sun': {
        'Mon': -1,
        'Tue': 0,
        'Wed': -1,
        'Thu': -2,
        'Fri': -3,
        'Sat': -4,
        'Sun': 0
    },
    'Wed_Thu': {
        'Mon': -4,
        'Tue': -5,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': -3
    },
    'Wed_Fri': {
        'Mon': -3,
        'Tue': -4,
        'Wed': 0,
        'Thu': -1,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Wed_Sat': {
        'Mon': -2,
        'Tue': -3,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': 0,
        'Sun': -1
    },
    'Wed_Sun': {
        'Mon': -1,
        'Tue': -2,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': -3,
        'Sun': 0
    },

    # 1-day scenarios
    'Only_Mon': {
        'Mon': 0,
        'Tue': -1,
        'Wed': -2,
        'Thu': -3,
        'Fri': -4,
        'Sat': -5,
        'Sun': -6
    },
    'Only_Tue': {
        'Mon': -6,
        'Tue': 0,
        'Wed': -1,
        'Thu': -2,
        'Fri': -3,
        'Sat': -4,
        'Sun': -5
    },
    'Only_Wed': {
        'Mon': -5,
        'Tue': -6,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': -3,
        'Sun': -4
    },
    'Only_Thu': {
        'Mon': -4,
        'Tue': -5,
        'Wed': -6,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': -3
    },
    'Only_Fri': {
        'Mon': -3,
        'Tue': -4,
        'Wed': -5,
        'Thu': -6,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Only_Sat': {
        'Mon': -2,
        'Tue': -3,
        'Wed': -4,
        'Thu': -5,
        'Fri': -6,
        'Sat': 0,
        'Sun': -1

    },
    'Only_Sun': {
        'Mon': -1,
        'Tue': -2,
        'Wed': -3,
        'Thu': -4,
        'Fri': -5,
        'Sat': -6,
        'Sun': 0

    }
}

scenarios = {
        "5 days delivery scenario": [
            "Mon_Tue_Wed_Thu_Fri", "Mon_Tue_Wed_Thu_Sat",
            "Mon_Tue_Wed_Fri_Sat", "Mon_Tue_Thu_Fri_Sat",
             "Mon_Wed_Thu_Fri_Sat",
            "Tue_Wed_Thu_Fri_Sat"
        ],
        "4 days delivery scenario": [
            "Mon_Tue_Wed_Thu", "Mon_Tue_Wed_Fri", "Mon_Tue_Wed_Sat",
            "Mon_Tue_Thu_Fri", "Mon_Tue_Thu_Sat", "Mon_Wed_Thu_Fri",
            "Mon_Wed_Thu_Sat", "Tue_Wed_Thu_Fri", "Tue_Wed_Thu_Sat"
            
        ],
        "3 days delivery scenario": [
            "Mon_Tue_Wed", "Mon_Tue_Thu", "Mon_Tue_Fri", "Mon_Tue_Sat",
            "Mon_Wed_Thu", "Mon_Wed_Fri", "Mon_Wed_Sat", 
            "Tue_Wed_Thu", "Tue_Wed_Fri", "Tue_Wed_Sat"
        ],
        "2 days delivery scenario": [
            "Mon_Tue", "Mon_Wed", "Mon_Thu", "Mon_Fri", "Mon_Sat",
            "Tue_Wed", "Tue_Thu", "Tue_Fri", "Tue_Sat",
            "Wed_Thu", "Wed_Fri", "Wed_Sat",
        ],
        "1 day delivery scenario": [
            "Only_Mon", "Only_Tue", "Only_Wed", "Only_Thu", "Only_Fri", "Only_Sat",
        ]
    }
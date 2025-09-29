import itertools

def format_spec_name(spec_list):
    if len(spec_list) == 1:
        return spec_list[0]
    elif len(spec_list) == 2:
        return f"{spec_list[0]} and {spec_list[1]}"
    else:
        return f"{', '.join(spec_list[:-1])}, and {spec_list[-1]}"

def gen_specs(k=3, embedding_model_name="default", observables=False):
    pc_specifications = {"plain logit": {}}

    # Base parameter list: price + PC1 to PCk
    params = ["price"] + [f"PC{i}" for i in range(1, k + 1)] if not observables else ["price", "pages", "year", "genre"]

    # Generate all non-empty combinations
    for r in range(1, len(params) + 1):
        for combo in itertools.combinations(params, r):
            spec_name = format_spec_name(combo)
            if not observables:
                pc_specifications[spec_name] = {
                                                f"{embedding_model_name}_{param.lower()}" 
                                                if param != "price" else param: "n"
                                                for param in combo
                                            }
            else:
                pc_specifications[spec_name] = {}
                for param in combo:
                    if param == "genre":
                        pc_specifications[spec_name][f"{param}_mystery"] = "n"
                        pc_specifications[spec_name][f"{param}_scifi"] = "n"
                    else:
                        pc_specifications[spec_name][param] = "n"

    return pc_specifications
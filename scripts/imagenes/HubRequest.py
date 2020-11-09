from sentinelhub import SHConfig

# In case you put the credentials into the configuration file you can leave this unchanged

CLIENT_ID = '01a88c6d-3398-476e-8684-8bb2ba74e44b'
CLIENT_SECRET = '|5bm$MLO;VJFN$^#^O9[u0DFR8^7cs&#^#PU++ZU'
config = SHConfig()

if CLIENT_ID and CLIENT_SECRET:
    config.sh_client_id = CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET

if config.sh_client_id == '' or config.sh_client_secret == '':
    print("Warning! To use Sentinel Hub services, please provide the credentials (client ID and client secret).")
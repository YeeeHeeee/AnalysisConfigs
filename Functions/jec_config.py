JECversions = {
    '2016_PreVFP': {
        'MC': 'Summer19UL16APV_V7_MC',
        'Data': {
            'B': 'Summer19UL16APV_RunBCD_V7_DATA',
            'C': 'Summer19UL16APV_RunBCD_V7_DATA',
            'D': 'Summer19UL16APV_RunBCD_V7_DATA',
            'E': 'Summer19UL16APV_RunEF_V7_DATA',
            'F': 'Summer19UL16APV_RunEF_V7_DATA',
        },
    },
    '2016_PostVFP': {
        'MC': 'Summer19UL16_V7_MC',
        'Data': {
            'F': 'Summer19UL16_RunFGH_V7_DATA',
            'G': 'Summer19UL16_RunFGH_V7_DATA',
            'H': 'Summer19UL16_RunFGH_V7_DATA',
        },
    },
    '2017': {
        'MC': 'Summer19UL17_V5_MC',
        'Data': {
            'B': 'Summer19UL17_RunB_V5_DATA',
            'C': 'Summer19UL17_RunC_V5_DATA',
            'D': 'Summer19UL17_RunD_V5_DATA',
            'E': 'Summer19UL17_RunE_V5_DATA',
            'F': 'Summer19UL17_RunF_V5_DATA',
        },
    },
    '2018': {
        'MC': 'Summer19UL18_V5_MC',
        'Data': {
            'A': 'Summer19UL18_RunA_V5_DATA',
            'B': 'Summer19UL18_RunB_V5_DATA',
            'C': 'Summer19UL18_RunC_V5_DATA',
            'D': 'Summer19UL18_RunD_V5_DATA',
        },
    },
    '2022_preEE' : {
        'MC': 'Summer22_22Sep2023_V2_MC',
        'Data': {
            'C': 'Summer22_22Sep2023_RunCD_V2_DATA',
            'D': 'Summer22_22Sep2023_RunCD_V2_DATA',
        },
    },
    '2022_postEE' : {
        'MC': 'Summer22EE_22Sep2023_V2_MC',
        'Data': {
            'E': 'Summer22EE_22Sep2023_RunE_V2_DATA',
            'F': 'Summer22EE_22Sep2023_RunF_V2_DATA',
            'G': 'Summer22EE_22Sep2023_RunG_V2_DATA',
        },
    },
    '2023_preBPix' : {
        'MC': 'Summer23Prompt23_V2_MC',
        'Data': {
            'Cv1': 'Summer23Prompt23_V2_DATA',
            'Cv2': 'Summer23Prompt23_V2_DATA',
            'Cv3': 'Summer23Prompt23_V2_DATA',
            'Cv4': 'Summer23Prompt23_V2_DATA',
        },
    },
    '2023_postBPix' : {
        'MC': 'Summer23BPixPrompt23_V3_MC',
        'Data': {
            'D': 'Summer23BPixPrompt23_V3_DATA',
        },
    },
}

JERversions = {
    '2016_PreVFP': {
        'MC': 'Summer20UL16APV_JRV3_MC',
        'Data': 'Summer20UL16APV_JRV3_DATA',
    },
    '2016_PostVFP': {
        'MC': 'Summer20UL16_JRV3_MC', 
        'Data': 'Summer20UL16_JRV3_DATA'
    },
    '2017': {
        'MC': 'Summer19UL17_JRV2_MC', 
        'Data': 'Summer19UL17_JRV2_DATA'
    },
    '2018': {
        'MC': 'Summer19UL18_JRV2_MC', 
        'Data': 'Summer19UL18_JRV2_DATA'
    },
    '2022_preEE': {
        'MC': 'Summer22_22Sep2023_JRV1_MC',
        'Data': 'Summer22_22Sep2023_JRV1_DATA',
    },
    '2022_postEE': {
        'MC': 'Summer22EE_22Sep2023_JRV1_MC',
        'Data': 'Summer22EE_22Sep2023_JRV1_DATA',
    },
    '2023_preBPix': {
        'MC': 'Summer23Prompt23_RunCv1234_JRV1_MC',
        'Data': 'Summer23Prompt23_RunCv1234_JRV1_DATA',
    },
    '2023_postBPix': {
        'MC': 'Summer23BPixPrompt23_RunD_JRV1_MC',
        'Data': 'Summer23BPixPrompt23_RunD_JRV1_DATA',
    },
}

JECjsonFiles = {
    '2016_PreVFP': {
        'AK4': '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2016preVFP_UL/jet_jerc.json.gz',
        'AK8': '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2016preVFP_UL/fatJet_jerc.json.gz',
    },
    '2016_PostVFP': {
        'AK4': '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2016postVFP_UL/jet_jerc.json.gz',
        'AK8': '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2016postVFP_UL/fatJet_jerc.json.gz',
    },
    '2017': {
        'AK4': '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2017_UL/jet_jerc.json.gz',
        'AK8': '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2017_UL/fatJet_jerc.json.gz',
    },
    '2018': {
        'AK4': '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2018_UL/jet_jerc.json.gz',
        'AK8': '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2018_UL/fatJet_jerc.json.gz',
    },
    '2022_preEE': {
        'AK4': '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2022_Summer22/jet_jerc.json.gz',
        'AK8': '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2022_Summer22/fatJet_jerc.json.gz',
    },
    '2022_postEE': {
        'AK4': '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2022_Summer22EE/jet_jerc.json.gz',
        'AK8': '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2022_Summer22EE/fatJet_jerc.json.gz',
    },
    '2023_preBPix': {
        'AK4': '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2023_Summer23/jet_jerc.json.gz',
        'AK8': '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2023_Summer23/fatJet_jerc.json.gz',
    },
    '2023_postBPix': {
        'AK4': '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2023_Summer23BPix/jet_jerc.json.gz',
        'AK8': '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2023_Summer23BPix/fatJet_jerc.json.gz',
    }
}
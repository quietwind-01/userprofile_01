package cn.itcast.userprofile.platform.service.impl;

import cn.itcast.up.common.OozieParam;
import cn.itcast.up.common.OozieUtils;
import cn.itcast.userprofile.platform.bean.dto.ModelDto;
import cn.itcast.userprofile.platform.service.Engine;
import org.springframework.stereotype.Service;

import java.util.Properties;

@Service
public class EngineImpl implements Engine {
    @Override
    public String startModel(ModelDto model) {
        //注意:之前在命令行中演示的是通过加载配置文件使用Oozie调度执行任务
        //而现在要是使用Oozie提供的API来调度执行任务
        //那么不管何种方式,配置的信息总是需要的,如调度周期,main方法,jar包位置...
        OozieParam param =  new OozieParam(model.getId(),
                model.getMainClass(),
                model.getPath(),
                model.getArgs(),
                ModelDto.Schedule.formatTime(model.getSchedule().getStartTime()),
                ModelDto.Schedule.formatTime(model.getSchedule().getEndTime()));

        //根据参数生成配置文件Properties对象
        Properties properties = OozieUtils.genProperties(param);

        //上传配置到HDFS(如coordinator.xml/workflow.xml)
        OozieUtils.uploadConfig(model.getId());

        //保留一份Properties方便后续如果出错可以查看
        OozieUtils.store(model.getId(),properties);

        //运行任务
        String jobId = OozieUtils.start(properties);

        return jobId;
    }

    @Override
    public void stopModel(ModelDto model) {
        //注意:前面开启任务的时候将任务的id保存到了name中,所以停止任务的时候使用name也就是jobid来停止
        OozieUtils.stop(model.getName());
    }
}

 function todimsearch(){//预搜索
		$("table").empty();
	 	var dimname=$("[name='dimname']").val();
        if(dimname==""||dimname==null){
			alert("请输入预搜索关键字！");
			return false;
		}
		$.get('/dimsearch/'+dimname,
			function(dimresult){
				$("select").empty();
				$("select").append("<option>===请选择一个选项===</option>");
				for(i in dimresult){
					$("select").append("<option value='"+dimresult[i]+"'>"+dimresult[i]+"</option>");
				}
			},'json');
	}
	function search(){//搜索
	 	var selectName=$("[name='selectName']").val();
        if(selectName==""||selectName==null){
			alert("请选择搜索项再搜索！");
			return false;
		}
		debugger
		var bl="";
		$("[name='type']:checked").each(function(){
			bl+=","+$(this).val();
		});
		var type=bl.substring(1).split(",");
		if(type[0]==""){
			type[0]="1";
		}
		var searchUrl='/search/'+selectName+'/'+type;
		$.get('/search/'+selectName+'/'+type,
			function(result){
				debugger
				$("table").empty();
				var thhonor="<th>荣耀相似度</th>";
				if("undefined"==typeof(result[0].荣耀)){
					thhonor="";
				}
				var thage="<th>年龄相似度</th>";
				if("undefined"==typeof(result[0].年龄)){
					thage="";
				}
				var thall="<th>总相似度</th>";
				if("undefined"==typeof(result[0].Sum_weight)){
					thall="";
				}
				$("table").append("<tr><th>姓名</th>"+thhonor+""+thage+""+thall+"</tr>");
				if("1"==type[0]){
						$("table").append("<tr><td>"+result[0].knowledge_node+"</td></tr>");
					}
				for(var j=1;j<=result.length;j++){
					var obj=result[j-1];
					var tdperson=obj.person;
					if("1"==type[0]){
						tdperson=obj.sub_knowledge_node;
					}
					var tdhonor="";
					if("undefined"==typeof(obj.荣耀)){
						tdhonor="";
					}else{
						tdhonor="<td>"+obj.荣耀+"</td>";
					}
					var tdage="";
					if("undefined"==typeof(obj.年龄)){
						tdage="";
					}else{
						tdage="<td>"+obj.年龄+"</td>";
					}
					var tdall="";
					if("undefined"==typeof(obj.Sum_weight)){
						tdall="";
					}else{
						tdall="<td>"+obj.Sum_weight+"</td>";
					}
					$("table").append("<tr><td>"+tdperson+"</td>"+tdhonor+""+tdage+""+tdall+"</tr>");
				}
			},'json');
	}